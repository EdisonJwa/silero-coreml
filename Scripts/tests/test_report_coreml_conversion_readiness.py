from __future__ import annotations

import builtins
import contextlib
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


MODULE_NAME = "report_coreml_conversion_readiness"
UPSTREAM_RELEASE_TAG = "v6.2.1"
UPSTREAM_COMMIT = "7e30209a3e901f9842f81b225f3e93d8199902b1"
SILERO_ONNX_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


class FakeDim:
    def __init__(self, value: int | None = None, param: str = "") -> None:
        self._has_dim_value = value is not None
        self._has_dim_param = bool(param)
        self.dim_value = 0 if value is None else value
        self.dim_param = param

    def HasField(self, name: str) -> bool:
        if name == "dim_value":
            return self._has_dim_value
        if name == "dim_param":
            return self._has_dim_param
        return False


class FakeShape:
    def __init__(self, dims: list[int | str]) -> None:
        self.dim = [FakeDim(value=dim) if isinstance(dim, int) else FakeDim(param=dim) for dim in dims]


class FakeTensorType:
    def __init__(self, dims: list[int | str], elem_type: int = 1) -> None:
        self.elem_type = elem_type
        self.shape = FakeShape(dims)

    def HasField(self, name: str) -> bool:
        return name == "shape"


class FakeType:
    def __init__(self, dims: list[int | str], elem_type: int = 1) -> None:
        self.tensor_type = FakeTensorType(dims, elem_type=elem_type)

    def HasField(self, name: str) -> bool:
        return name == "tensor_type"


class FakeValueInfo:
    def __init__(self, name: str, dims: list[int | str], elem_type: int = 1) -> None:
        self.name = name
        self.type = FakeType(dims, elem_type=elem_type)


class FakeNode:
    def __init__(self, op_type: str) -> None:
        self.op_type = op_type


class FakeOpsetImport:
    def __init__(self, domain: str, version: int) -> None:
        self.domain = domain
        self.version = version


class FakeGraph:
    def __init__(self) -> None:
        self.input = [
            FakeValueInfo("input", [1, 576]),
            FakeValueInfo("state", [2, 1, 128]),
        ]
        self.output = [
            FakeValueInfo("output", [1, 1]),
            FakeValueInfo("stateN", [2, 1, 128]),
        ]
        self.initializer = [object(), object()]
        self.node = [
            FakeNode("MatMul"),
            FakeNode("Add"),
            FakeNode("Add"),
            FakeNode("Sigmoid"),
        ]


class FakeModel:
    def __init__(self) -> None:
        self.graph = FakeGraph()
        self.opset_import = [
            FakeOpsetImport("", 17),
            FakeOpsetImport("ai.onnx.ml", 3),
        ]


class FakeTensorProto:
    class DataType:
        @staticmethod
        def Name(value: int) -> str:
            return {
                1: "FLOAT",
            }.get(value, f"UNKNOWN_{value}")


class ReportCoreMLConversionReadinessTests(unittest.TestCase):
    def assert_dependency_payload_has_onnxruntime(self, payload: dict[str, object]) -> None:
        dependency_availability = payload.get("dependency_availability")
        self.assertIsInstance(dependency_availability, dict)
        dependency_availability = cast(dict[str, Any], dependency_availability)
        self.assertIn("onnxruntime", dependency_availability)

        onnxruntime_payload = dependency_availability["onnxruntime"]
        self.assertIsInstance(onnxruntime_payload, dict)
        onnxruntime_payload = cast(dict[str, Any], onnxruntime_payload)
        self.assertIn("available", onnxruntime_payload)
        self.assertIn("version", onnxruntime_payload)
        self.assertIsInstance(onnxruntime_payload["available"], bool)
        self.assertTrue(
            onnxruntime_payload["version"] is None or isinstance(onnxruntime_payload["version"], str)
        )

    def write_manifest(self, root: Path) -> Path:
        manifest_path = root / "manifest.json"
        payload = {
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
                    "sha256": "jit-sha-placeholder",
                },
                {
                    "path": "src/silero_vad/data/silero_vad.onnx",
                    "filename": "silero_vad.onnx",
                    "raw_url": "https://example.invalid/silero_vad.onnx",
                    "sha256": SILERO_ONNX_SHA256,
                },
            ],
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return manifest_path

    def import_reporter(self):
        sys.modules.pop(MODULE_NAME, None)
        importlib.invalidate_caches()
        try:
            return importlib.import_module(MODULE_NAME)
        except ModuleNotFoundError as exc:
            if exc.name == MODULE_NAME:
                self.fail(
                    "Expected future module 'report_coreml_conversion_readiness' to exist once implemented."
                )
            raise

    def test_package_import_path_uses_relative_conversion_helpers(self) -> None:
        sys.modules.pop("Scripts.report_coreml_conversion_readiness", None)
        importlib.invalidate_caches()

        reporter = importlib.import_module("Scripts.report_coreml_conversion_readiness")

        self.assertEqual(reporter.__package__, "Scripts")
        self.assertTrue(hasattr(reporter, "report_readiness"))

    @contextlib.contextmanager
    def missing_onnx_import(self):
        original_import = builtins.__import__
        saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "onnx" or name.startswith("onnx.")
        }
        for name in list(saved_modules):
            sys.modules.pop(name, None)

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "onnx" or name.startswith("onnx."):
                raise ModuleNotFoundError("No module named 'onnx'")
            return original_import(name, globals, locals, fromlist, level)

        try:
            with mock.patch("builtins.__import__", new=guarded_import):
                yield
        finally:
            for name, module in saved_modules.items():
                sys.modules[name] = module

    def build_fake_onnx_module(self) -> types.SimpleNamespace:
        fake_model = FakeModel()

        def load_model(_path: object) -> FakeModel:
            return fake_model

        return types.SimpleNamespace(
            TensorProto=FakeTensorProto,
            load=load_model,
            load_model=load_model,
        )

    def test_default_mode_missing_local_artifact_reports_skipped_and_creates_no_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root)
            artifacts_dir = root / "local-artifacts"
            report_path = root / "reports" / "readiness-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()

            with self.missing_onnx_import():
                reporter = self.import_reporter()
                exit_code = reporter.report_readiness(
                    manifest_path=manifest_path,
                    artifacts_dir=artifacts_dir,
                    source_artifact_name="silero_vad.jit",
                    out_path=out_path,
                    report_path=report_path,
                    dry_run=False,
                    printer=lambda line: print(line, file=stdout),
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(artifacts_dir.exists(), "readiness reporting should not create artifact directories")
            self.assertFalse(report_path.exists(), "readiness reporting should not write a report file")
            self.assertFalse(report_path.parent.exists(), "readiness reporting should not create report directories")
            self.assertFalse(out_path.exists(), "readiness reporting should not create conversion output")
            self.assertFalse(out_path.parent.exists(), "readiness reporting should not create output directories")

            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["dry_run"])
            self.assertEqual(payload["conversion_status"], "planned")
            self.assert_dependency_payload_has_onnxruntime(payload)
            self.assertEqual(payload["onnx_introspection"]["status"], "skipped_non_onnx_source")
            self.assertEqual(payload["target_contract"]["input"]["shape"], [1, 576])
            self.assertEqual(payload["target_contract"]["state"]["name"], "state_in")
            self.assertEqual(payload["target_contract"]["state"]["shape"], [2, 1, 128])

    def test_main_reports_missing_dependency_when_onnx_package_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root)
            artifacts_dir = root / "local-artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "silero_vad.onnx").write_bytes(b"placeholder-local-model")
            report_path = root / "reports" / "readiness-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()

            with self.missing_onnx_import():
                reporter = self.import_reporter()
                with contextlib.redirect_stdout(stdout):
                    exit_code = reporter.main(
                        [
                            "--manifest",
                            str(manifest_path),
                            "--artifacts-dir",
                            str(artifacts_dir),
                            "--report",
                            str(report_path),
                            "--out",
                            str(out_path),
                            "--source-artifact",
                            "silero_vad.onnx",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertFalse(report_path.exists(), "readiness reporting should stay read-only without onnx")
            self.assertFalse(report_path.parent.exists(), "readiness reporting should not create report directories")
            self.assertFalse(out_path.exists(), "readiness reporting should not create conversion output")
            self.assertFalse(out_path.parent.exists(), "readiness reporting should not create output directories")

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["conversion_status"], "planned")
            self.assert_dependency_payload_has_onnxruntime(payload)
            self.assertEqual(payload["onnx_introspection"]["status"], "skipped_missing_dependency")
            self.assertEqual(payload["expected_local_source_path"], str((artifacts_dir / "silero_vad.onnx").resolve()))

    def test_report_readiness_includes_mocked_onnx_introspection_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root)
            artifacts_dir = root / "local-artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "silero_vad.onnx").write_bytes(b"placeholder-local-model")
            report_path = root / "reports" / "readiness-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()

            with mock.patch.dict(sys.modules, {"onnx": self.build_fake_onnx_module()}):
                reporter = self.import_reporter()
                exit_code = reporter.report_readiness(
                    manifest_path=manifest_path,
                    artifacts_dir=artifacts_dir,
                    source_artifact_name="silero_vad.onnx",
                    out_path=out_path,
                    report_path=report_path,
                    dry_run=False,
                    printer=lambda line: print(line, file=stdout),
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(report_path.exists(), "readiness reporting should not write a report file")
            self.assertFalse(report_path.parent.exists(), "readiness reporting should not create report directories")
            self.assertFalse(out_path.exists(), "readiness reporting should not create conversion output")
            self.assertFalse(out_path.parent.exists(), "readiness reporting should not create output directories")

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["conversion_status"], "planned")
            self.assert_dependency_payload_has_onnxruntime(payload)
            self.assertEqual(payload["target_contract"]["input"]["shape"], [1, 576])
            self.assertEqual(payload["target_contract"]["state"]["name"], "state_in")
            self.assertEqual(payload["target_contract"]["state"]["shape"], [2, 1, 128])

            introspection = payload["onnx_introspection"]
            self.assertEqual(
                [{"name": item["name"], "shape": item["shape"]} for item in introspection["inputs"]],
                [
                    {"name": "input", "shape": [1, 576]},
                    {"name": "state", "shape": [2, 1, 128]},
                ],
            )
            self.assertEqual(
                [{"name": item["name"], "shape": item["shape"]} for item in introspection["outputs"]],
                [
                    {"name": "output", "shape": [1, 1]},
                    {"name": "stateN", "shape": [2, 1, 128]},
                ],
            )
            self.assertEqual(
                introspection["opset_imports"],
                [
                    {"domain": "", "version": 17},
                    {"domain": "ai.onnx.ml", "version": 3},
                ],
            )
            self.assertEqual(introspection["initializer_count"], 2)
            self.assertEqual(
                introspection["node_type_counts"],
                {
                    "Add": 2,
                    "MatMul": 1,
                    "Sigmoid": 1,
                },
            )

    def test_dry_run_readiness_creates_no_artifact_report_or_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root)
            artifacts_dir = root / "local-artifacts"
            report_path = root / "reports" / "readiness-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()

            with self.missing_onnx_import():
                reporter = self.import_reporter()
                exit_code = reporter.report_readiness(
                    manifest_path=manifest_path,
                    artifacts_dir=artifacts_dir,
                    source_artifact_name="silero_vad.jit",
                    out_path=out_path,
                    report_path=report_path,
                    dry_run=True,
                    printer=lambda line: print(line, file=stdout),
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(artifacts_dir.exists(), "dry-run should not create artifact directories")
            self.assertFalse(report_path.exists(), "dry-run should not write a report file")
            self.assertFalse(report_path.parent.exists(), "dry-run should not create report directories")
            self.assertFalse(out_path.exists(), "dry-run should not create conversion output")
            self.assertFalse(out_path.parent.exists(), "dry-run should not create output directories")

            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["conversion_status"], "planned")
            self.assert_dependency_payload_has_onnxruntime(payload)
            self.assertEqual(payload["onnx_introspection"]["status"], "skipped_non_onnx_source")


if __name__ == "__main__":
    unittest.main()
