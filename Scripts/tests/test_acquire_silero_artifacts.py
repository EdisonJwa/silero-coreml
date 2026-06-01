from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

acquire = importlib.import_module("acquire_silero_artifacts")

UPSTREAM_COMMIT = "7e30209a3e901f9842f81b225f3e93d8199902b1"


class FakeResponse:
    def __init__(self, payload: bytes):
        self._stream = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def fake_urlopen_factory(payloads: dict[str, bytes]):
    def fake_urlopen(url: str):
        try:
            return FakeResponse(payloads[url])
        except KeyError as exc:
            raise AssertionError(f"unexpected URL requested: {url}") from exc

    return fake_urlopen


class AcquireSileroArtifactsTests(unittest.TestCase):
    def write_manifest(
        self,
        root: Path,
        *,
        payload: bytes,
        filename: str = "artifact.bin",
        sha256: str | None = None,
        upstream_commit: str = UPSTREAM_COMMIT,
    ) -> Path:
        manifest_path = root / "manifest.json"
        raw_url = f"https://example.invalid/{filename}"
        artifact = {
            "path": f"src/silero_vad/data/{filename}",
            "filename": filename,
            "size_bytes": len(payload),
            "blob_sha": "blob-sha-placeholder",
            "raw_url": raw_url,
        }
        if sha256 is not None:
            artifact["sha256"] = sha256

        manifest = {
            "upstream": {
                "commit": upstream_commit,
            },
            "artifacts": [artifact],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_dry_run_prints_plans_and_creates_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root, payload=b"dry-run-payload")
            dest_root = root / "artifacts"
            stdout = io.StringIO()

            exit_code = acquire.acquire_artifacts(
                manifest_path=manifest_path,
                dest_root=dest_root,
                dry_run=True,
                printer=lambda line: print(line, file=stdout),
            )

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn(f"Dry run: upstream_commit={UPSTREAM_COMMIT}", output)
            self.assertIn("PLAN https://example.invalid/artifact.bin", output)
            self.assertFalse(dest_root.exists(), "dry-run should not create destination directories")

    def test_download_writes_artifacts_sha256sums_and_report(self) -> None:
        payload = b"official-silero-artifact"
        expected_sha256 = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(
                root,
                payload=payload,
                filename="silero_vad.onnx",
                sha256=expected_sha256,
            )
            dest_root = root / "artifacts"
            stdout = io.StringIO()

            exit_code = acquire.acquire_artifacts(
                manifest_path=manifest_path,
                dest_root=dest_root,
                dry_run=False,
                urlopen=fake_urlopen_factory({"https://example.invalid/silero_vad.onnx": payload}),
                printer=lambda line: print(line, file=stdout),
            )

            self.assertEqual(exit_code, 0)

            commit_root = dest_root / UPSTREAM_COMMIT
            artifact_path = commit_root / "silero_vad.onnx"
            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact_path.read_bytes(), payload)

            sha256sums = (commit_root / "SHA256SUMS").read_text(encoding="utf-8")
            self.assertEqual(sha256sums, f"{expected_sha256}  silero_vad.onnx\n")

            report = json.loads((commit_root / "acquisition-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["upstream_commit"], UPSTREAM_COMMIT)
            self.assertEqual(len(report["artifacts"]), 1)
            artifact = report["artifacts"][0]
            self.assertEqual(artifact["path"], "src/silero_vad/data/silero_vad.onnx")
            self.assertEqual(artifact["filename"], "silero_vad.onnx")
            self.assertEqual(artifact["size_bytes"], len(payload))
            self.assertEqual(artifact["blob_sha"], "blob-sha-placeholder")
            self.assertEqual(artifact["raw_url"], "https://example.invalid/silero_vad.onnx")
            self.assertEqual(artifact["expected_sha256"], expected_sha256)
            self.assertEqual(artifact["sha256"], expected_sha256)

    def test_sha256_mismatch_fails_and_does_not_leave_partial_file(self) -> None:
        payload = b"official-silero-artifact"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(
                root,
                payload=payload,
                filename="silero_vad.onnx",
                sha256="0" * 64,
            )
            dest_root = root / "artifacts"
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr), mock.patch(
                "acquire_silero_artifacts.urllib.request.urlopen",
                fake_urlopen_factory({"https://example.invalid/silero_vad.onnx": payload}),
            ):
                exit_code = acquire.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--dest",
                        str(dest_root),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("Downloaded SHA-256 mismatch for silero_vad.onnx", stderr.getvalue())

            commit_root = dest_root / UPSTREAM_COMMIT
            self.assertTrue(commit_root.exists(), "commit directory may exist after download setup")
            self.assertFalse((commit_root / "silero_vad.onnx").exists())
            self.assertFalse(any(commit_root.glob("*.tmp")), "temporary files should be cleaned up")

    def test_size_mismatch_fails_and_does_not_leave_partial_file(self) -> None:
        declared_payload = b"1234567890"
        downloaded_payload = b"12345"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(root, payload=declared_payload, filename="silero_vad.jit")
            dest_root = root / "artifacts"
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr), mock.patch(
                "acquire_silero_artifacts.urllib.request.urlopen",
                fake_urlopen_factory({"https://example.invalid/silero_vad.jit": downloaded_payload}),
            ):
                exit_code = acquire.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--dest",
                        str(dest_root),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("Downloaded size mismatch for silero_vad.jit", stderr.getvalue())

            commit_root = dest_root / UPSTREAM_COMMIT
            self.assertTrue(commit_root.exists(), "commit directory may exist after download setup")
            self.assertFalse((commit_root / "silero_vad.jit").exists())
            self.assertFalse(any(commit_root.glob("*.tmp")), "temporary files should be cleaned up")

    def test_manifest_commit_must_be_lowercase_hex_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_manifest(
                root,
                payload=b"payload",
                upstream_commit="../../escape",
            )

            with self.assertRaisesRegex(acquire.AcquisitionError, "40-character lowercase hex"):
                acquire.load_manifest(manifest_path)

    def test_commit_directory_must_remain_under_destination_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dest_root = root / "artifacts"

            with self.assertRaisesRegex(acquire.AcquisitionError, "under destination root"):
                acquire.commit_directory(dest_root, "..")


if __name__ == "__main__":
    unittest.main()
