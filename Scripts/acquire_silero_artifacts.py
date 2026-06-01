#!/usr/bin/env python3
"""Acquire pinned official Silero VAD artifacts from a provenance manifest.

This repository intentionally does not commit upstream model assets. Instead,
the manifest in `Docs/ModelProvenance/` records the exact `snakers4/silero-vad`
commit and raw artifact URLs that may be fetched locally for validation work.

Examples:

    python3 Scripts/acquire_silero_artifacts.py --dry-run
    python3 Scripts/acquire_silero_artifacts.py --dest .artifacts/silero-vad
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "Docs" / "ModelProvenance" / "silero-vad-v6.2.1.json"
DEFAULT_DEST = REPO_ROOT / ".artifacts" / "silero-vad"
CHUNK_SIZE = 1024 * 1024
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class AcquisitionError(Exception):
    pass


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    filename: str
    size_bytes: int
    blob_sha: str
    raw_url: str
    sha256: str | None = None


@dataclass(frozen=True)
class ManifestRecord:
    upstream_commit: str
    artifacts: tuple[ArtifactRecord, ...]


@dataclass(frozen=True)
class DownloadResult:
    artifact: ArtifactRecord
    destination: Path
    sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or download pinned official Silero VAD artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to a Silero VAD provenance manifest JSON file.",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help="Directory root where artifacts should be stored locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without creating files or directories.",
    )
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest = Path(args.manifest).expanduser().resolve()
    if not manifest.exists():
        parser.error(f"manifest path does not exist: {manifest}")
    if manifest.suffix.lower() != ".json":
        parser.error(f"manifest path must be a JSON file: {manifest}")

    dest = Path(args.dest).expanduser().resolve()
    args.manifest = manifest
    args.dest = dest
    return args


def load_manifest(manifest_path: Path) -> ManifestRecord:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AcquisitionError(f"Manifest is not valid JSON: {manifest_path}: {exc}") from exc

    upstream = payload.get("upstream")
    if not isinstance(upstream, dict):
        raise AcquisitionError("Manifest must contain an 'upstream' object.")

    upstream_commit = upstream.get("commit")
    if not isinstance(upstream_commit, str) or not upstream_commit:
        raise AcquisitionError("Manifest upstream.commit must be a non-empty string.")
    if COMMIT_SHA_PATTERN.fullmatch(upstream_commit) is None:
        raise AcquisitionError("Manifest upstream.commit must be a 40-character lowercase hex commit SHA.")

    artifacts_payload = payload.get("artifacts")
    if not isinstance(artifacts_payload, list) or not artifacts_payload:
        raise AcquisitionError("Manifest artifacts must be a non-empty list.")

    artifacts: list[ArtifactRecord] = []
    for index, artifact_payload in enumerate(artifacts_payload, start=1):
        if not isinstance(artifact_payload, dict):
            raise AcquisitionError(f"Manifest artifact #{index} must be an object.")
        try:
            path = artifact_payload["path"]
            filename = artifact_payload["filename"]
            size_bytes = artifact_payload["size_bytes"]
            blob_sha = artifact_payload["blob_sha"]
            raw_url = artifact_payload["raw_url"]
        except KeyError as exc:
            raise AcquisitionError(
                f"Manifest artifact #{index} is missing required field: {exc.args[0]}"
            ) from exc

        if not isinstance(path, str) or not path:
            raise AcquisitionError(f"Manifest artifact #{index} path must be a non-empty string.")
        if not isinstance(filename, str) or not filename:
            raise AcquisitionError(
                f"Manifest artifact #{index} filename must be a non-empty string."
            )
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise AcquisitionError(
                f"Manifest artifact #{index} size_bytes must be a non-negative integer."
            )
        if not isinstance(blob_sha, str) or not blob_sha:
            raise AcquisitionError(
                f"Manifest artifact #{index} blob_sha must be a non-empty string."
            )
        if not isinstance(raw_url, str) or not raw_url:
            raise AcquisitionError(
                f"Manifest artifact #{index} raw_url must be a non-empty string."
            )
        sha256 = artifact_payload.get("sha256")
        if sha256 is not None and (not isinstance(sha256, str) or not sha256):
            raise AcquisitionError(
                f"Manifest artifact #{index} sha256 must be a non-empty string when present."
            )
        if Path(filename).name != filename:
            raise AcquisitionError(
                f"Manifest artifact #{index} filename must not contain path separators: {filename}"
            )
        if Path(path).name != filename:
            raise AcquisitionError(
                f"Manifest artifact #{index} filename does not match path basename: {filename}"
            )

        artifacts.append(
            ArtifactRecord(
                path=path,
                filename=filename,
                size_bytes=size_bytes,
                blob_sha=blob_sha,
                raw_url=raw_url,
                sha256=sha256,
            )
        )

    return ManifestRecord(upstream_commit=upstream_commit, artifacts=tuple(artifacts))


def commit_directory(dest_root: Path, upstream_commit: str) -> Path:
    commit_root = (dest_root / upstream_commit).resolve()
    dest_root_resolved = dest_root.resolve()
    if commit_root != dest_root_resolved and dest_root_resolved not in commit_root.parents:
        raise AcquisitionError("Resolved commit artifact directory must remain under destination root.")
    return commit_root


def plan_downloads(
    manifest: ManifestRecord,
    dest_root: Path,
) -> list[tuple[ArtifactRecord, Path]]:
    commit_root = commit_directory(dest_root, manifest.upstream_commit)
    return [
        (artifact, commit_root / artifact.filename)
        for artifact in manifest.artifacts
    ]


def print_dry_run(
    *,
    manifest_path: Path,
    manifest: ManifestRecord,
    dest_root: Path,
    plans: list[tuple[ArtifactRecord, Path]],
    printer: Callable[[str], None],
) -> None:
    printer(f"Dry run: manifest={manifest_path}")
    printer(f"Dry run: upstream_commit={manifest.upstream_commit}")
    printer(f"Dry run: destination_root={dest_root}")
    for artifact, destination in plans:
        printer(
            "PLAN "
            f"{artifact.raw_url} -> {destination} "
            f"size_bytes={artifact.size_bytes} blob_sha={artifact.blob_sha}"
            + (f" sha256={artifact.sha256}" if artifact.sha256 else "")
        )


def download_artifact(
    *,
    artifact: ArtifactRecord,
    destination: Path,
    urlopen: Callable[..., object],
) -> DownloadResult:
    hasher = hashlib.sha256()
    bytes_written = 0
    temp_file: Any | None = None
    temp_path: Path | None = None

    try:
        response_context = cast(Any, urlopen(artifact.raw_url))
        with response_context as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=destination.parent,
                prefix=f".{artifact.filename}.",
                suffix=".tmp",
            ) as handle:
                temp_file = handle
                temp_path = Path(handle.name)
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    hasher.update(chunk)
                    bytes_written += len(chunk)

        if bytes_written != artifact.size_bytes:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            raise AcquisitionError(
                f"Downloaded size mismatch for {artifact.filename}: "
                f"expected {artifact.size_bytes} bytes, got {bytes_written}"
            )

        if temp_path is None:
            raise AcquisitionError(f"Failed to create a temporary file for {artifact.filename}")

        actual_sha256 = hasher.hexdigest()
        if artifact.sha256 is not None and actual_sha256 != artifact.sha256:
            if temp_path.exists():
                temp_path.unlink()
            raise AcquisitionError(
                f"Downloaded SHA-256 mismatch for {artifact.filename}: "
                f"expected {artifact.sha256}, got {actual_sha256}"
            )

        temp_path.replace(destination)
        return DownloadResult(
            artifact=artifact,
            destination=destination,
            sha256=actual_sha256,
        )
    except AcquisitionError:
        raise
    except Exception as exc:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise AcquisitionError(
            f"Failed to download {artifact.raw_url} to {destination}: {exc}"
        ) from exc
    finally:
        if temp_file is not None:
            try:
                temp_file.close()
            except Exception:
                pass


def write_sha256sums(commit_root: Path, results: list[DownloadResult]) -> Path:
    sha256_path = commit_root / "SHA256SUMS"
    lines = [f"{result.sha256}  {result.artifact.filename}" for result in results]
    sha256_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_path


def write_acquisition_report(
    commit_root: Path,
    manifest: ManifestRecord,
    results: list[DownloadResult],
) -> Path:
    report_path = commit_root / "acquisition-report.json"
    payload = {
        "upstream_commit": manifest.upstream_commit,
        "artifacts": [
            {
                "path": result.artifact.path,
                "filename": result.artifact.filename,
                "size_bytes": result.artifact.size_bytes,
                "blob_sha": result.artifact.blob_sha,
                "expected_sha256": result.artifact.sha256,
                "raw_url": result.artifact.raw_url,
                "sha256": result.sha256,
            }
            for result in results
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def acquire_artifacts(
    *,
    manifest_path: Path,
    dest_root: Path,
    dry_run: bool,
    urlopen: Callable[..., object] | None = None,
    printer: Callable[[str], None] = print,
) -> int:
    if urlopen is None:
        urlopen = urllib.request.urlopen

    manifest = load_manifest(manifest_path)
    plans = plan_downloads(manifest, dest_root)

    if dry_run:
        print_dry_run(
            manifest_path=manifest_path,
            manifest=manifest,
            dest_root=dest_root,
            plans=plans,
            printer=printer,
        )
        return 0

    commit_root = commit_directory(dest_root, manifest.upstream_commit)
    commit_root.mkdir(parents=True, exist_ok=True)

    results: list[DownloadResult] = []
    for artifact, destination in plans:
        printer(f"Downloading {artifact.raw_url} -> {destination}")
        result = download_artifact(
            artifact=artifact,
            destination=destination,
            urlopen=urlopen,
        )
        printer(
            f"Verified {artifact.filename}: size_bytes={artifact.size_bytes} sha256={result.sha256}"
        )
        results.append(result)

    sha256_path = write_sha256sums(commit_root, results)
    report_path = write_acquisition_report(commit_root, manifest, results)
    printer(f"Wrote {sha256_path}")
    printer(f"Wrote {report_path}")
    return 0


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    try:
        return acquire_artifacts(
            manifest_path=args.manifest,
            dest_root=args.dest,
            dry_run=args.dry_run,
        )
    except AcquisitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: filesystem failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
