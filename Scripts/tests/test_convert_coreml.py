from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


MODULE_NAME = "convert_coreml"
UPSTREAM_RELEASE_TAG = "v6.2.1"
UPSTREAM_COMMIT = "7e30209a3e901f9842f81b225f3e93d8199902b1"


class ConvertCoreMLTests(unittest.TestCase):
    def import_convert(self):
        sys.modules.pop(MODULE_NAME, None)
        importlib.invalidate_caches()
        return importlib.import_module(MODULE_NAME)

    def write_manifest(self, root: Path, sha256: str | None, *, filename: str = "silero_vad.jit") -> Path:
        manifest_path = root / "manifest.json"
        suffix = Path(filename).suffix
        artifact: dict[str, str] = {
            "path": f"src/silero_vad/data/{filename}",
            "filename": filename,
            "format": suffix,
            "raw_url": f"https://example.invalid/{filename}",
        }
        if sha256 is not None:
            artifact["sha256"] = sha256
        payload = {
            "upstream": {
                "release_tag": UPSTREAM_RELEASE_TAG,
                "commit": UPSTREAM_COMMIT,
                "source_of_truth": f"https://example.invalid/tree/{UPSTREAM_RELEASE_TAG}",
            },
            "artifacts": [artifact],
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return manifest_path

    def conversion_patches(self, convert):
        stack = contextlib.ExitStack()
        probe = sys.modules[convert.build_probe_payload.__module__]
        stack.enter_context(mock.patch.object(probe, "collect_dependency_checks", return_value={}))
        stack.enter_context(
            mock.patch.object(probe, "inspect_onnx_interface", return_value={"status": "skipped_test_stub"})
        )
        stack.enter_context(
            mock.patch.object(probe, "run_onnxruntime_smoke", return_value={"status": "skipped_test_stub"})
        )
        stack.enter_context(
            mock.patch.object(
                probe,
                "probe_coremltools_route",
                return_value={"status": "unsupported", "blocker": "direct ONNX unsupported"},
            )
        )
        return stack

    def test_dry_run_prints_decision_and_creates_no_files_or_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_bytes = b"local-jit-placeholder"
            manifest_path = self.write_manifest(root, hashlib.sha256(model_bytes).hexdigest())
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "silero_vad.jit").write_bytes(model_bytes)
            report_path = root / "reports" / "conversion-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()
            convert = self.import_convert()

            with self.conversion_patches(convert):
                exit_code = convert.convert_coreml(
                    manifest_path=manifest_path,
                    artifacts_dir=artifacts_dir,
                    source_artifact_name="silero_vad.jit",
                    out_path=out_path,
                    report_path=report_path,
                    dry_run=True,
                    printer=lambda line: print(line, file=stdout),
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(report_path.exists(), "dry-run should not write a report")
            self.assertFalse(report_path.parent.exists(), "dry-run should not create report directories")
            self.assertFalse(out_path.exists(), "dry-run should not create a CoreML asset")
            self.assertFalse(out_path.parent.exists(), "dry-run should not create output directories")

            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["conversion_status"], "blocked_prerequisites")
            self.assertEqual(payload["attempted_route"]["name"], "torchscript_explicit_state_conversion")
            self.assertEqual(payload["attempted_route"]["status"], "implemented")
            self.assertFalse(payload["attempted_route"]["supported"])
            self.assertTrue(payload["attempted_route"]["preserves_direct_onnx_evidence"])
            self.assertEqual(payload["direct_onnx_conversion"]["status"], "unsupported")
            self.assertEqual(payload["torchscript_explicit_state_conversion"]["status"], "implemented")
            self.assertFalse(payload["output_written"])
            self.assertFalse(payload["writes_report"])
            self.assertEqual(payload["source_artifact_verification"]["sha256_status"], "match")
            self.assertEqual(payload["upstream"]["commit"], UPSTREAM_COMMIT)
            self.assertTrue(
                any(blocker["component"] == "dependency:torch" for blocker in payload["blockers"])
            )

    def test_blocked_non_dry_run_writes_report_but_no_output_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_bytes = b"local-jit-placeholder"
            manifest_path = self.write_manifest(root, hashlib.sha256(model_bytes).hexdigest())
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "silero_vad.jit").write_bytes(model_bytes)
            report_path = root / "reports" / "conversion-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()
            stderr = io.StringIO()
            convert = self.import_convert()

            with self.conversion_patches(convert):
                exit_code = convert.convert_coreml(
                    manifest_path=manifest_path,
                    artifacts_dir=artifacts_dir,
                    source_artifact_name="silero_vad.jit",
                    out_path=out_path,
                    report_path=report_path,
                    dry_run=False,
                    printer=lambda line: print(line, file=stdout),
                    error_printer=lambda line: print(line, file=stderr),
                )

            self.assertEqual(exit_code, convert.BLOCKED_EXIT_CODE)
            self.assertTrue(report_path.exists())
            self.assertFalse(out_path.exists(), "blocked conversion must not write .mlpackage")
            self.assertFalse(out_path.parent.exists(), "blocked conversion must not create output directory")
            self.assertIn("blocked: CoreML conversion prerequisites are not satisfied", stderr.getvalue())
            self.assertIn(str(out_path), stderr.getvalue())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            printed = json.loads(stdout.getvalue())
            self.assertEqual(report, printed)
            self.assertFalse(report["dry_run"])
            self.assertEqual(report["conversion_status"], "blocked_prerequisites")
            self.assertEqual(report["output_path"], str(out_path.resolve()))
            self.assertEqual(report["report_path"], str(report_path.resolve()))
            self.assertEqual(report["target_contract"]["sample_rate_hz"], 16000)
            self.assertEqual(report["source_artifact_verification"]["sha256_status"], "match")
            self.assertTrue(
                any(
                    blocker["component"] == "dependency:torch"
                    for blocker in report["blockers"]
                )
            )
            self.assertTrue(
                any(
                    blocker["component"] == "direct_onnx_conversion"
                    for blocker in report["probe_blockers"]
                )
            )

    def test_missing_source_artifact_is_reported_as_blocked_with_exit_code_2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root, "0" * 64)
            artifacts_dir = root / "missing-artifacts"
            report_path = root / "reports" / "conversion-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()
            stderr = io.StringIO()
            convert = self.import_convert()

            with self.conversion_patches(convert):
                exit_code = convert.convert_coreml(
                    manifest_path=manifest_path,
                    artifacts_dir=artifacts_dir,
                    source_artifact_name="silero_vad.jit",
                    out_path=out_path,
                    report_path=report_path,
                    dry_run=False,
                    printer=lambda line: print(line, file=stdout),
                    error_printer=lambda line: print(line, file=stderr),
                )

            self.assertEqual(exit_code, convert.BLOCKED_EXIT_CODE)
            self.assertTrue(report_path.exists())
            self.assertFalse(out_path.exists())
            self.assertFalse(artifacts_dir.exists(), "converter should not create missing artifact dirs")

            report = json.loads(report_path.read_text(encoding="utf-8"))
            verification = cast(dict[str, Any], report["source_artifact_verification"])
            self.assertEqual(verification["status"], "missing_artifact")
            self.assertEqual(verification["sha256_status"], "not_checked")
            self.assertTrue(
                any(blocker["component"] == "source_artifact" for blocker in report["blockers"])
            )
            self.assertEqual(report["conversion_status"], "blocked_prerequisites")
            self.assertIn("did not write", stderr.getvalue())

    def test_manifest_missing_source_hash_is_clear_blocker_in_decision_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root, None)
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "silero_vad.jit").write_bytes(b"local-jit-placeholder")
            report_path = root / "reports" / "conversion-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()
            convert = self.import_convert()

            with self.conversion_patches(convert):
                exit_code = convert.convert_coreml(
                    manifest_path=manifest_path,
                    artifacts_dir=artifacts_dir,
                    source_artifact_name="silero_vad.jit",
                    out_path=out_path,
                    report_path=report_path,
                    dry_run=False,
                    printer=lambda line: print(line, file=stdout),
                    error_printer=lambda _line: None,
                )

            self.assertEqual(exit_code, convert.BLOCKED_EXIT_CODE)
            self.assertTrue(report_path.exists())
            self.assertFalse(out_path.exists())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            verification = cast(dict[str, Any], report["source_artifact_verification"])
            self.assertEqual(verification["status"], "available")
            self.assertEqual(verification["expected_sha256"], None)
            self.assertEqual(verification["sha256_status"], "manifest_sha_missing")
            self.assertEqual(report["conversion_status"], "blocked_prerequisites")
            self.assertTrue(
                any(
                    blocker["component"] == "source_artifact"
                    and "SHA-256" in blocker["message"]
                    for blocker in report["blockers"]
                )
            )
            self.assertTrue(any(blocker["component"] == "source_artifact" for blocker in report["blockers"]))

    def test_main_blocked_route_returns_clear_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_bytes = b"local-jit-placeholder"
            manifest_path = self.write_manifest(root, hashlib.sha256(model_bytes).hexdigest())
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "silero_vad.jit").write_bytes(model_bytes)
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            report_path = root / "reports" / "conversion-report.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            convert = self.import_convert()

            with self.conversion_patches(convert):
                with contextlib.redirect_stdout(stdout):
                    with contextlib.redirect_stderr(stderr):
                        exit_code = convert.main(
                            [
                                "--manifest",
                                str(manifest_path),
                                "--artifacts-dir",
                                str(artifacts_dir),
                                "--source-artifact",
                                "silero_vad.jit",
                                "--out",
                                str(out_path),
                                "--report",
                                str(report_path),
                            ]
                        )

            self.assertEqual(exit_code, convert.BLOCKED_EXIT_CODE)
            self.assertTrue(report_path.exists())
            self.assertFalse(out_path.exists())
            self.assertEqual(json.loads(stdout.getvalue())["conversion_status"], "blocked_prerequisites")
            self.assertIn("blocked:", stderr.getvalue())

    def test_supported_jit_route_writes_output_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_bytes = b"local-jit-placeholder"
            manifest_path = self.write_manifest(root, hashlib.sha256(model_bytes).hexdigest())
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "silero_vad.jit").write_bytes(model_bytes)
            report_path = root / "reports" / "conversion-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()
            convert = self.import_convert()

            def fake_convert_torchscript_to_coreml(*, source_path: Path, out_path: Path):
                self.assertEqual(source_path, (artifacts_dir / "silero_vad.jit").resolve())
                out_path.mkdir(parents=True)
                (out_path / "Manifest.json").write_text("{}", encoding="utf-8")
                return {
                    "status": "converted",
                    "source_path": str(source_path),
                    "output_path": str(out_path),
                    "trace": "torch.jit.trace",
                    "coremltools_convert_to": "mlprogram",
                    "minimum_deployment_target": "iOS15",
                    "compute_units": "CPU_ONLY",
                }

            dependency_checks = {
                "torch": {"available": True, "version": "2.7.0", "status": "available", "blocker": None},
                "coremltools": {"available": True, "version": "9.0", "status": "available", "blocker": None},
            }

            with self.conversion_patches(convert):
                with mock.patch.object(convert, "collect_conversion_dependency_checks", return_value=dependency_checks):
                    with mock.patch.object(
                        convert,
                        "convert_torchscript_to_coreml",
                        side_effect=fake_convert_torchscript_to_coreml,
                    ):
                        exit_code = convert.convert_coreml(
                            manifest_path=manifest_path,
                            artifacts_dir=artifacts_dir,
                            source_artifact_name="silero_vad.jit",
                            out_path=out_path,
                            report_path=report_path,
                            dry_run=False,
                            printer=lambda line: print(line, file=stdout),
                        )

            self.assertEqual(exit_code, 0)
            self.assertTrue(out_path.exists())
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            printed = json.loads(stdout.getvalue())
            self.assertEqual(report, printed)
            self.assertEqual(report["conversion_status"], "converted")
            self.assertEqual(report["overall_status"], "converted")
            self.assertTrue(report["output_written"])
            self.assertEqual(report["attempted_route"]["status"], "converted")
            self.assertTrue(report["attempted_route"]["supported"])
            self.assertEqual(report["blockers"], [])

    def test_convert_torchscript_to_coreml_uses_expected_trace_and_coreml_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "silero_vad.jit"
            source_path.write_bytes(b"fake-jit")
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            convert = self.import_convert()
            calls: dict[str, Any] = {}

            class FakeModule:
                def __init__(self) -> None:
                    self.was_evaled = False

                def eval(self):
                    self.was_evaled = True
                    return self

                def register_buffer(self, name: str, value: object) -> None:
                    calls.setdefault("registered_buffers", []).append((name, value))
                    setattr(self, name, value)

            class FakeWeight:
                def __init__(self, name: str) -> None:
                    self.name = name

                def detach(self):
                    calls.setdefault("detached_weights", []).append(self.name)
                    return self

                def clone(self):
                    calls.setdefault("cloned_weights", []).append(self.name)
                    return f"cloned:{self.name}"

            class FakeRNN:
                def __init__(self) -> None:
                    self.weight_ih = FakeWeight("weight_ih")
                    self.weight_hh = FakeWeight("weight_hh")
                    self.bias_ih = FakeWeight("bias_ih")
                    self.bias_hh = FakeWeight("bias_hh")

            class FakeDecoder:
                def __init__(self) -> None:
                    self.rnn = FakeRNN()
                    self.decoder = lambda value: ("output", value)

            class FakeInternalModel:
                def __init__(self) -> None:
                    self.stft = lambda value: ("features", value)
                    self.encoder = lambda value: ("encoded", value)
                    self.decoder = FakeDecoder()

            class FakeJitModel(FakeModule):
                def __init__(self) -> None:
                    super().__init__()
                    self._model = FakeInternalModel()

            class FakeTorchModuleBase:
                def __init__(self) -> None:
                    self.was_evaled = False

                def eval(self):
                    self.was_evaled = True
                    return self

                def register_buffer(self, name: str, value: object) -> None:
                    calls.setdefault("registered_buffers", []).append((name, value))
                    setattr(self, name, value)

            class FakeNoGrad:
                def __enter__(self):
                    calls["no_grad_entered"] = True

                def __exit__(self, exc_type, exc, traceback):
                    calls["no_grad_exited"] = True

            class FakeJit:
                @staticmethod
                def load(path: str, *, map_location: object):
                    calls["jit_load"] = {"path": path, "map_location": map_location}
                    return FakeJitModel()

                @staticmethod
                def trace(wrapper: object, example_args: tuple[object, object], *, strict: bool):
                    calls["trace"] = {
                        "wrapper": wrapper,
                        "example_args": example_args,
                        "strict": strict,
                    }
                    return "traced-model"

                @staticmethod
                def freeze(model: object):
                    calls["freeze"] = model
                    return "frozen-model"

            def fake_zeros(shape: tuple[int, ...], *, dtype: object):
                return {"shape": shape, "dtype": dtype}

            fake_torch = types.SimpleNamespace(
                nn=types.SimpleNamespace(Module=FakeTorchModuleBase),
                jit=FakeJit,
                device=lambda name: f"device:{name}",
                float32="float32",
                zeros=fake_zeros,
                no_grad=FakeNoGrad,
                set_num_threads=lambda value: calls.setdefault("num_threads", value),
            )

            class FakeTensorType:
                def __init__(self, *, name: str, shape: tuple[int, ...] | None = None) -> None:
                    self.name = name
                    self.shape = shape

            class FakeConvertedModel:
                def save(self, path: str) -> None:
                    calls["save_path"] = path
                    Path(path).mkdir(parents=True, exist_ok=True)

            def fake_convert(model: object, **kwargs: object) -> FakeConvertedModel:
                calls["coreml_convert"] = {"model": model, **kwargs}
                return FakeConvertedModel()

            fake_coremltools = types.SimpleNamespace(
                TensorType=FakeTensorType,
                convert=fake_convert,
                target=types.SimpleNamespace(iOS15="iOS15"),
                ComputeUnit=types.SimpleNamespace(CPU_ONLY="CPU_ONLY"),
            )

            def fake_import_module(name: str):
                if name == "torch":
                    return fake_torch
                if name == "coremltools":
                    return fake_coremltools
                return importlib.import_module(name)

            with mock.patch.object(convert.importlib, "import_module", side_effect=fake_import_module):
                result = convert.convert_torchscript_to_coreml(
                    source_path=source_path,
                    out_path=out_path,
                )

            self.assertEqual(calls["num_threads"], 1)
            self.assertEqual(calls["jit_load"], {"path": str(source_path), "map_location": "device:cpu"})
            self.assertTrue(calls["trace"]["strict"])
            self.assertEqual(calls["freeze"], "traced-model")
            self.assertEqual(
                calls["detached_weights"],
                ["weight_ih", "weight_hh", "bias_ih", "bias_hh"],
            )
            self.assertEqual(
                calls["registered_buffers"],
                [
                    ("weight_ih", "cloned:weight_ih"),
                    ("weight_hh", "cloned:weight_hh"),
                    ("bias_ih", "cloned:bias_ih"),
                    ("bias_hh", "cloned:bias_hh"),
                ],
            )
            self.assertEqual(calls["trace"]["example_args"][0], {"shape": (1, 576), "dtype": "float32"})
            self.assertEqual(calls["trace"]["example_args"][1], {"shape": (2, 1, 128), "dtype": "float32"})
            coreml_call = calls["coreml_convert"]
            self.assertEqual(coreml_call["model"], "frozen-model")
            self.assertEqual(coreml_call["convert_to"], "mlprogram")
            self.assertEqual(coreml_call["minimum_deployment_target"], "iOS15")
            self.assertEqual(coreml_call["compute_units"], "CPU_ONLY")
            self.assertEqual(
                [(item.name, item.shape) for item in coreml_call["inputs"]],
                [("input", (1, 576)), ("state_in", (2, 1, 128))],
            )
            self.assertEqual(
                [(item.name, item.shape) for item in coreml_call["outputs"]],
                [("output", None), ("stateN", None)],
            )
            self.assertEqual(calls["save_path"], str(out_path))
            self.assertTrue(out_path.exists())
            self.assertEqual(result["status"], "converted")
            self.assertEqual(result["trace"], "torch.jit.trace+torch.jit.freeze")


if __name__ == "__main__":
    unittest.main()
