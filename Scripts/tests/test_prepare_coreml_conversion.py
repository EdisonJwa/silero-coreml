from __future__ import annotations

import contextlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

prepare = importlib.import_module("prepare_coreml_conversion")


UPSTREAM_RELEASE_TAG = "v6.2.1"
UPSTREAM_COMMIT = "7e30209a3e901f9842f81b225f3e93d8199902b1"
SILERO_ONNX_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_JIT_SHA256 = "jit-sha-placeholder"


class PrepareCoreMLConversionTests(unittest.TestCase):
    def write_manifest(self, root: Path, *, upstream_commit: str = UPSTREAM_COMMIT) -> Path:
        manifest_path = root / "manifest.json"
        payload = {
            "upstream": {
                "release_tag": UPSTREAM_RELEASE_TAG,
                "commit": upstream_commit,
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

    def test_manifest_rejects_non_sha_upstream_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root, upstream_commit="../../outside")

            with self.assertRaisesRegex(
                prepare.PreparationError,
                "upstream.commit must be a 40-character lowercase hexadecimal SHA",
            ):
                prepare.load_manifest(manifest_path)

    def test_dry_run_prints_plan_and_creates_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root)
            artifacts_dir = root / "local-artifacts"
            report_path = root / "reports" / "conversion-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()

            exit_code = prepare.prepare_conversion(
                manifest_path=manifest_path,
                artifacts_dir=artifacts_dir,
                source_artifact_name="silero_vad.onnx",
                out_path=out_path,
                report_path=report_path,
                dry_run=True,
                printer=lambda line: print(line, file=stdout),
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse(artifacts_dir.exists(), "dry-run should not create artifact directories")
            self.assertFalse(report_path.exists(), "dry-run should not write a report")
            self.assertFalse(out_path.exists(), "dry-run should not create an output placeholder")

            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["selected_source_artifact"]["filename"], "silero_vad.onnx")
            self.assertEqual(payload["output_path"], str(out_path.resolve()))

    def test_default_source_selection_prefers_silero_vad_jit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root)
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = prepare.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--artifacts-dir",
                        str(root / "artifacts"),
                        "--report",
                        str(root / "report.json"),
                        "--out",
                        str(root / "SileroVAD.mlpackage"),
                        "--dry-run",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["selected_source_artifact"]["filename"], "silero_vad.jit")
            self.assertEqual(payload["selected_source_artifact"]["sha256"], SILERO_JIT_SHA256)

    def test_report_mode_fails_when_expected_local_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root)
            artifacts_dir = root / "local-artifacts"
            stderr = io.StringIO()
            report_path = root / "reports" / "conversion-report.json"

            with contextlib.redirect_stderr(stderr):
                exit_code = prepare.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--artifacts-dir",
                        str(artifacts_dir),
                        "--report",
                        str(report_path),
                        "--out",
                        str(root / "coreml" / "SileroVAD.mlpackage"),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("Expected local source artifact does not exist", stderr.getvalue())
            self.assertFalse(report_path.exists(), "report should not be written when source is missing")

    def test_report_mode_writes_expected_payload_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root)
            artifacts_dir = root / "local-artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            local_source = artifacts_dir / "silero_vad.onnx"
            local_source.write_bytes(b"placeholder-local-model")
            report_path = root / "reports" / "conversion-report.json"
            out_path = root / "coreml" / "SileroVAD.mlpackage"
            stdout = io.StringIO()

            exit_code = prepare.prepare_conversion(
                manifest_path=manifest_path,
                artifacts_dir=artifacts_dir,
                source_artifact_name="silero_vad.onnx",
                out_path=out_path,
                report_path=report_path,
                dry_run=False,
                printer=lambda line: print(line, file=stdout),
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(report_path.exists())
            self.assertFalse(out_path.exists(), "prepare step should not perform conversion output writes")

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["dry_run"])
            self.assertEqual(payload["conversion_status"], "planned")
            self.assertEqual(payload["upstream"]["release_tag"], UPSTREAM_RELEASE_TAG)
            self.assertEqual(payload["upstream"]["commit"], UPSTREAM_COMMIT)
            self.assertEqual(payload["selected_source_artifact"]["path"], "src/silero_vad/data/silero_vad.onnx")
            self.assertEqual(payload["selected_source_artifact"]["filename"], "silero_vad.onnx")
            self.assertEqual(payload["selected_source_artifact"]["sha256"], SILERO_ONNX_SHA256)
            self.assertEqual(payload["expected_local_source_path"], str(local_source.resolve()))
            self.assertEqual(payload["output_path"], str(out_path.resolve()))

            target_contract = payload["target_contract"]
            self.assertEqual(target_contract["sample_rate_hz"], 16000)
            self.assertEqual(target_contract["input"]["shape"], [1, 576])
            self.assertEqual(target_contract["state"]["name"], "state_in")
            self.assertEqual(target_contract["state"]["shape"], [2, 1, 128])
            self.assertEqual(target_contract["host_context_size"], 64)
            self.assertEqual(target_contract["chunk_size"], 512)
            self.assertEqual(target_contract["outputs"]["probability"]["name"], "output")
            self.assertEqual(target_contract["outputs"]["next_state"]["shape"], [2, 1, 128])

            packaging = payload["packaging_recommendation"]
            self.assertEqual(packaging["conversion_output"], ".mlpackage")
            self.assertEqual(packaging["future_swiftpm_runtime_resource"], ".mlmodelc")


if __name__ == "__main__":
    unittest.main()
