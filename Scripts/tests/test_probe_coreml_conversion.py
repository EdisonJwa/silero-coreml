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


MODULE_NAME = "probe_coreml_conversion"
UPSTREAM_RELEASE_TAG = "v6.2.1"
UPSTREAM_COMMIT = "7e30209a3e901f9842f81b225f3e93d8199902b1"


class ProbeCoreMLConversionTests(unittest.TestCase):
    def import_probe(self):
        sys.modules.pop(MODULE_NAME, None)
        importlib.invalidate_caches()
        return importlib.import_module(MODULE_NAME)

    def write_manifest(self, root: Path, sha256: str | None) -> Path:
        manifest_path = root / "manifest.json"
        artifact = {
            "path": "src/silero_vad/data/silero_vad.onnx",
            "filename": "silero_vad.onnx",
            "raw_url": "https://example.invalid/silero_vad.onnx",
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

    def build_payload_without_ml_imports(
        self,
        *,
        root: Path,
        manifest_path: Path,
        artifacts_dir: Path,
    ) -> dict[str, Any]:
        probe = self.import_probe()
        with mock.patch.object(probe, "collect_dependency_checks", return_value={}):
            with mock.patch.object(
                probe,
                "inspect_onnx_interface",
                return_value={"status": "skipped_test_stub"},
            ):
                with mock.patch.object(
                    probe,
                    "run_onnxruntime_smoke",
                    return_value={"status": "skipped_test_stub"},
                ):
                    with mock.patch.object(
                        probe,
                        "probe_coremltools_route",
                        return_value={"status": "unsupported", "blocker": "direct ONNX unsupported"},
                    ):
                        return cast(dict[str, Any], probe.build_probe_payload(
                            manifest_path=manifest_path,
                            artifacts_dir=artifacts_dir,
                            source_artifact_name="silero_vad.onnx",
                            out_path=root / "coreml" / "SileroVAD.mlpackage",
                            report_path=None,
                        ))

    def test_hash_match_reports_manifest_sha_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_bytes = b"local-onnx-placeholder"
            expected_sha = hashlib.sha256(model_bytes).hexdigest()
            manifest_path = self.write_manifest(root, expected_sha)
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "silero_vad.onnx").write_bytes(model_bytes)

            payload = self.build_payload_without_ml_imports(
                root=root,
                manifest_path=manifest_path,
                artifacts_dir=artifacts_dir,
            )

            verification = cast(dict[str, Any], payload["source_artifact_verification"])
            self.assertEqual(verification["status"], "available")
            self.assertEqual(verification["sha256_status"], "match")
            self.assertEqual(verification["actual_sha256"], expected_sha)
            self.assertIsNone(verification["blocker"])
            self.assertEqual(payload["torchscript_explicit_state_conversion"]["status"], "implemented")
            self.assertTrue(payload["torchscript_explicit_state_conversion"]["supported"])
            self.assertFalse(payload["writes_report"])
            self.assertIsNone(payload["report_path"])

    def test_hash_mismatch_is_structured_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root, "0" * 64)
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "silero_vad.onnx").write_bytes(b"different-model-bytes")

            payload = self.build_payload_without_ml_imports(
                root=root,
                manifest_path=manifest_path,
                artifacts_dir=artifacts_dir,
            )

            verification = cast(dict[str, Any], payload["source_artifact_verification"])
            self.assertEqual(verification["status"], "sha256_mismatch")
            self.assertEqual(verification["sha256_status"], "mismatch")
            self.assertEqual(payload["overall_status"], "blocked")
            self.assertIn(
                {"component": "source_artifact", "message": verification["blocker"]},
                payload["blockers"],
            )

    def test_hash_mismatch_skips_onnx_parsing_and_runtime_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root, "0" * 64)
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "silero_vad.onnx").write_bytes(b"tampered-model-bytes")
            probe = self.import_probe()

            with mock.patch.object(probe, "collect_dependency_checks", return_value={}):
                with mock.patch.object(probe, "inspect_onnx_interface") as inspect_mock:
                    with mock.patch.object(probe, "run_onnxruntime_smoke") as smoke_mock:
                        with mock.patch.object(
                            probe,
                            "probe_coremltools_route",
                            return_value={"status": "unsupported", "blocker": "direct ONNX unsupported"},
                        ):
                            payload = probe.build_probe_payload(
                                manifest_path=manifest_path,
                                artifacts_dir=artifacts_dir,
                                source_artifact_name="silero_vad.onnx",
                                out_path=root / "coreml" / "SileroVAD.mlpackage",
                                report_path=None,
                            )

            inspect_mock.assert_not_called()
            smoke_mock.assert_not_called()
            self.assertEqual(payload["onnx_interface"]["status"], "skipped_unverified_artifact")
            self.assertEqual(payload["onnxruntime_smoke"]["status"], "skipped_unverified_artifact")
            self.assertEqual(payload["overall_status"], "blocked")

    def test_missing_artifact_reports_blocked_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root, "0" * 64)
            artifacts_dir = root / "missing-artifacts"
            report_path = root / "reports" / "probe.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()
            probe = self.import_probe()

            with mock.patch.object(probe, "collect_dependency_checks", return_value={}):
                with mock.patch.object(
                    probe,
                    "probe_coremltools_route",
                    return_value={"status": "unsupported", "blocker": "direct ONNX unsupported"},
                ):
                    exit_code = probe.probe_coreml_conversion(
                        manifest_path=manifest_path,
                        artifacts_dir=artifacts_dir,
                        source_artifact_name="silero_vad.onnx",
                        out_path=out_path,
                        report_path=None,
                        printer=lambda line: print(line, file=stdout),
                    )

            self.assertEqual(exit_code, 0)
            self.assertFalse(artifacts_dir.exists())
            self.assertFalse(report_path.exists())
            self.assertFalse(out_path.exists())
            self.assertFalse(out_path.parent.exists())

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["overall_status"], "blocked")
            self.assertEqual(payload["source_artifact_verification"]["status"], "missing_artifact")
            self.assertEqual(payload["onnx_interface"]["status"], "skipped_unverified_artifact")
            self.assertEqual(payload["onnxruntime_smoke"]["status"], "skipped_unverified_artifact")
            self.assertFalse(payload["writes_report"])
            self.assertIsNone(payload["report_path"])

    def test_missing_dependencies_are_structured_blockers(self) -> None:
        probe = self.import_probe()

        with mock.patch.object(probe, "_module_available_without_import", return_value=False):
            with mock.patch.object(probe, "_distribution_version", return_value=None):
                dependency_checks = probe.collect_dependency_checks()

        for dependency_name in ("onnx", "onnxruntime", "coremltools", "torch", "numpy"):
            self.assertEqual(dependency_checks[dependency_name]["status"], "missing")
            self.assertFalse(dependency_checks[dependency_name]["available"])
            self.assertIn(dependency_name, dependency_checks[dependency_name]["blocker"])

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "silero_vad.onnx"
            source_path.write_bytes(b"placeholder")
            real_import_module = importlib.import_module

            def guarded_import(name: str, package: str | None = None):
                if name in {"onnx", "onnxruntime", "numpy"}:
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return real_import_module(name, package)

            with mock.patch.object(probe.importlib, "import_module", side_effect=guarded_import):
                onnx_interface = probe.inspect_onnx_interface(source_path)
                smoke = probe.run_onnxruntime_smoke(source_path)

        self.assertEqual(onnx_interface["status"], "skipped_missing_dependency")
        self.assertEqual(onnx_interface["dependency"], "onnx")
        self.assertIn("blocker", onnx_interface)
        self.assertEqual(smoke["status"], "skipped_missing_dependency")
        self.assertEqual(smoke["dependency"], "numpy")
        self.assertIn("blocker", smoke)

    def test_coremltools_route_reports_unsupported_shape_for_fake_version_9(self) -> None:
        probe = self.import_probe()
        fake_coremltools = types.SimpleNamespace(
            __version__="9.0",
            converters=types.SimpleNamespace(),
            convert=lambda model: model,
        )

        with mock.patch.object(probe, "_module_available_without_import", return_value=True):
            with mock.patch.object(probe, "_distribution_version", return_value="9.0"):
                with mock.patch.dict(sys.modules, {"coremltools": fake_coremltools}):
                    route = probe.probe_coremltools_route()

        self.assertEqual(route["status"], "unsupported")
        self.assertTrue(route["available"])
        self.assertEqual(route["version"], "9.0")
        self.assertEqual(route["direct_onnx_symbols"], [])
        self.assertIn("explicit-state PyTorch module", route["recommendation"])
        self.assertIn("torch.jit.trace", route["recommendation"])
        self.assertTrue(
            any("Python 3.13" in fact for fact in route["release_note_facts"]),
            route["release_note_facts"],
        )
        self.assertTrue(
            any("PyTorch 2.7" in fact for fact in route["release_note_facts"]),
            route["release_note_facts"],
        )
        self.assertTrue(
            any("state read/write" in fact for fact in route["release_note_facts"]),
            route["release_note_facts"],
        )
        self.assertTrue(
            any("do not explicitly mention ONNX" in fact for fact in route["release_note_facts"]),
            route["release_note_facts"],
        )

    def test_planned_torchscript_route_documents_explicit_state_jit_path(self) -> None:
        probe = self.import_probe()

        route = probe.planned_torchscript_route()

        self.assertEqual(route["status"], "implemented")
        self.assertTrue(route["supported"])
        self.assertIn("torch.jit.trace", route["recommendation"])
        self.assertIn("input,state_in -> output,stateN", route["target_contract"])
        self.assertIn("convert-pytorch", route["apple_reference"])

    def test_verified_jit_source_skips_onnx_specific_probes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_bytes = b"local-jit-placeholder"
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "upstream": {
                            "release_tag": UPSTREAM_RELEASE_TAG,
                            "commit": UPSTREAM_COMMIT,
                            "source_of_truth": f"https://example.invalid/tree/{UPSTREAM_RELEASE_TAG}",
                        },
                        "artifacts": [
                            {
                                "path": "src/silero_vad/data/silero_vad.jit",
                                "filename": "silero_vad.jit",
                                "raw_url": "https://example.invalid/silero_vad.jit",
                                "sha256": hashlib.sha256(model_bytes).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "silero_vad.jit").write_bytes(model_bytes)
            probe = self.import_probe()

            with mock.patch.object(probe, "collect_dependency_checks", return_value={}):
                with mock.patch.object(probe, "inspect_onnx_interface") as inspect_mock:
                    with mock.patch.object(probe, "run_onnxruntime_smoke") as smoke_mock:
                        with mock.patch.object(
                            probe,
                            "probe_coremltools_route",
                            return_value={"status": "unsupported", "blocker": "direct ONNX unsupported"},
                        ):
                            payload = probe.build_probe_payload(
                                manifest_path=manifest_path,
                                artifacts_dir=artifacts_dir,
                                source_artifact_name="silero_vad.jit",
                                out_path=root / "coreml" / "SileroVAD.mlpackage",
                                report_path=None,
                            )

            inspect_mock.assert_not_called()
            smoke_mock.assert_not_called()
            self.assertEqual(payload["source_artifact_verification"]["sha256_status"], "match")
            self.assertEqual(payload["onnx_interface"]["status"], "skipped_non_onnx_source")
            self.assertEqual(payload["onnxruntime_smoke"]["status"], "skipped_non_onnx_source")
            self.assertEqual(payload["torchscript_explicit_state_conversion"]["status"], "implemented")

    def test_main_default_prints_json_without_report_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root, "0" * 64)
            artifacts_dir = root / "missing-artifacts"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()
            probe = self.import_probe()

            with mock.patch.object(probe, "collect_dependency_checks", return_value={}):
                with mock.patch.object(
                    probe,
                    "probe_coremltools_route",
                    return_value={"status": "unsupported", "blocker": "direct ONNX unsupported"},
                ):
                    with contextlib.redirect_stdout(stdout):
                        exit_code = probe.main(
                            [
                                "--manifest",
                                str(manifest_path),
                                "--artifacts-dir",
                                str(artifacts_dir),
                                "--source-artifact",
                                "silero_vad.onnx",
                                "--out",
                                str(out_path),
                            ]
                        )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["selected_source_artifact"]["filename"], "silero_vad.onnx")
            self.assertEqual(payload["overall_status"], "blocked")
            self.assertIsNone(payload["report_path"])
            self.assertFalse(payload["writes_report"])
            self.assertFalse(out_path.exists())
            self.assertFalse(out_path.parent.exists())


if __name__ == "__main__":
    unittest.main()
