#!/usr/bin/env python3
"""Prepare a Silero VAD CoreML conversion plan without converting assets.

This repository intentionally stays asset-free. The provenance manifest in
`Docs/ModelProvenance/` pins the official upstream artifact metadata, while this
script formalizes the current best-known 16 kHz CoreML target contract.

Examples:

    python3 Scripts/prepare_coreml_conversion.py --dry-run
    python3 Scripts/prepare_coreml_conversion.py --report .artifacts/coreml/report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "Docs" / "ModelProvenance" / "silero-vad-v6.2.1.json"
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / ".artifacts" / "silero-vad"
DEFAULT_SOURCE_ARTIFACT = "silero_vad.jit"
DEFAULT_OUTPUT = REPO_ROOT / ".artifacts" / "coreml" / "SileroVAD.mlpackage"
DEFAULT_REPORT = REPO_ROOT / ".artifacts" / "coreml" / "SileroVAD.conversion-report.json"
UPSTREAM_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

TARGET_CONTRACT = {
    "sample_rate_hz": 16000,
    "input": {
        "name": "input",
        "shape": [1, 576],
        "description": "64 host-managed context samples + 512 fresh 16 kHz samples",
    },
    "state": {
        "name": "state_in",
        "shape": [2, 1, 128],
    },
    "host_context_size": 64,
    "chunk_size": 512,
    "outputs": {
        "probability": {
            "name": "output",
            "description": "Speech probability output",
        },
        "next_state": {
            "name": "stateN",
            "shape": [2, 1, 128],
        },
    },
}

PACKAGING_RECOMMENDATION = {
    "conversion_output": ".mlpackage",
    "future_swiftpm_runtime_resource": ".mlmodelc",
    "notes": [
        "Write the conversion output as an .mlpackage during offline conversion work.",
        "Prefer a validated .mlmodelc as the future SwiftPM runtime resource.",
        "Do not bundle model assets in git until validation and licensing review are complete.",
    ],
}


class PreparationError(Exception):
    pass


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    filename: str
    raw_url: str
    sha256: str | None


@dataclass(frozen=True)
class ManifestRecord:
    release_tag: str
    upstream_commit: str
    source_of_truth: str | None
    artifacts: tuple[ArtifactRecord, ...]


@dataclass(frozen=True)
class ConversionPaths:
    manifest_path: Path
    artifacts_dir: Path
    out_path: Path
    report_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a manifest-driven Silero VAD CoreML conversion plan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to a Silero VAD provenance manifest JSON file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory containing pinned local upstream artifacts. Defaults to .artifacts/silero-vad/<manifest upstream commit>.",
    )
    parser.add_argument(
        "--source-artifact",
        default=DEFAULT_SOURCE_ARTIFACT,
        help="Filename of the source artifact to prepare for conversion.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Planned CoreML conversion output path.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="Path where the JSON conversion report should be written in non-dry-run mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the JSON conversion plan without creating files or directories.",
    )
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        parser.error(f"manifest path does not exist: {manifest_path}")
    if manifest_path.suffix.lower() != ".json":
        parser.error(f"manifest path must be a JSON file: {manifest_path}")

    args.manifest = manifest_path
    args.artifacts_dir = (
        Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir is not None else None
    )
    args.out = Path(args.out).expanduser().resolve()
    args.report = Path(args.report).expanduser().resolve()

    if not isinstance(args.source_artifact, str) or not args.source_artifact:
        parser.error("source artifact must be a non-empty filename")
    if Path(args.source_artifact).name != args.source_artifact:
        parser.error(f"source artifact must not contain path separators: {args.source_artifact}")

    return args


def load_manifest(manifest_path: Path) -> ManifestRecord:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreparationError(f"Manifest is not valid JSON: {manifest_path}: {exc}") from exc

    upstream = payload.get("upstream")
    if not isinstance(upstream, dict):
        raise PreparationError("Manifest must contain an 'upstream' object.")

    release_tag = upstream.get("release_tag")
    if not isinstance(release_tag, str) or not release_tag:
        raise PreparationError("Manifest upstream.release_tag must be a non-empty string.")

    upstream_commit = upstream.get("commit")
    if not isinstance(upstream_commit, str) or not upstream_commit:
        raise PreparationError("Manifest upstream.commit must be a non-empty string.")
    if not UPSTREAM_COMMIT_PATTERN.fullmatch(upstream_commit):
        raise PreparationError("Manifest upstream.commit must be a 40-character lowercase hexadecimal SHA.")

    source_of_truth = upstream.get("source_of_truth")
    if source_of_truth is not None and (not isinstance(source_of_truth, str) or not source_of_truth):
        raise PreparationError(
            "Manifest upstream.source_of_truth must be a non-empty string when present."
        )

    artifacts_payload = payload.get("artifacts")
    if not isinstance(artifacts_payload, list) or not artifacts_payload:
        raise PreparationError("Manifest artifacts must be a non-empty list.")

    artifacts: list[ArtifactRecord] = []
    for index, artifact_payload in enumerate(artifacts_payload, start=1):
        if not isinstance(artifact_payload, dict):
            raise PreparationError(f"Manifest artifact #{index} must be an object.")

        try:
            path = artifact_payload["path"]
            filename = artifact_payload["filename"]
            raw_url = artifact_payload["raw_url"]
        except KeyError as exc:
            raise PreparationError(
                f"Manifest artifact #{index} is missing required field: {exc.args[0]}"
            ) from exc

        sha256 = artifact_payload.get("sha256")

        if not isinstance(path, str) or not path:
            raise PreparationError(f"Manifest artifact #{index} path must be a non-empty string.")
        if not isinstance(filename, str) or not filename:
            raise PreparationError(
                f"Manifest artifact #{index} filename must be a non-empty string."
            )
        if Path(filename).name != filename:
            raise PreparationError(
                f"Manifest artifact #{index} filename must not contain path separators: {filename}"
            )
        if Path(path).name != filename:
            raise PreparationError(
                f"Manifest artifact #{index} filename does not match path basename: {filename}"
            )
        if not isinstance(raw_url, str) or not raw_url:
            raise PreparationError(
                f"Manifest artifact #{index} raw_url must be a non-empty string."
            )
        if sha256 is not None and (not isinstance(sha256, str) or not sha256):
            raise PreparationError(
                f"Manifest artifact #{index} sha256 must be a non-empty string when present."
            )

        artifacts.append(
            ArtifactRecord(
                path=path,
                filename=filename,
                raw_url=raw_url,
                sha256=sha256,
            )
        )

    return ManifestRecord(
        release_tag=release_tag,
        upstream_commit=upstream_commit,
        source_of_truth=source_of_truth,
        artifacts=tuple(artifacts),
    )


def default_artifacts_dir(manifest: ManifestRecord) -> Path:
    return (DEFAULT_ARTIFACTS_ROOT / manifest.upstream_commit).resolve()


def resolve_paths(
    *,
    manifest_path: Path,
    manifest: ManifestRecord,
    artifacts_dir: Path | None,
    out_path: Path,
    report_path: Path,
) -> ConversionPaths:
    return ConversionPaths(
        manifest_path=manifest_path.resolve(),
        artifacts_dir=(artifacts_dir if artifacts_dir is not None else default_artifacts_dir(manifest)).resolve(),
        out_path=out_path.resolve(),
        report_path=report_path.resolve(),
    )


def select_source_artifact(manifest: ManifestRecord, filename: str) -> ArtifactRecord:
    for artifact in manifest.artifacts:
        if artifact.filename == filename:
            return artifact
    available = ", ".join(sorted(artifact.filename for artifact in manifest.artifacts))
    raise PreparationError(
        f"Source artifact '{filename}' not found in manifest. Available artifacts: {available}"
    )


def build_conversion_plan(
    *,
    manifest_path: Path,
    manifest: ManifestRecord,
    paths: ConversionPaths,
    source_artifact: ArtifactRecord,
    dry_run: bool,
) -> dict[str, Any]:
    expected_local_source_path = (paths.artifacts_dir / source_artifact.filename).resolve()
    return {
        "manifest_path": str(paths.manifest_path),
        "dry_run": dry_run,
        "upstream": {
            "release_tag": manifest.release_tag,
            "commit": manifest.upstream_commit,
            "source_of_truth": manifest.source_of_truth,
        },
        "selected_source_artifact": {
            "path": source_artifact.path,
            "filename": source_artifact.filename,
            "sha256": source_artifact.sha256,
            "raw_url": source_artifact.raw_url,
        },
        "expected_local_source_path": str(expected_local_source_path),
        "output_path": str(paths.out_path),
        "report_path": str(paths.report_path),
        "target_contract": TARGET_CONTRACT,
        "packaging_recommendation": PACKAGING_RECOMMENDATION,
        "conversion_status": "planned",
    }


def emit_json(payload: dict[str, Any], *, printer: Callable[[str], None]) -> None:
    printer(json.dumps(payload, indent=2, sort_keys=True))


def write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_conversion(
    *,
    manifest_path: Path,
    artifacts_dir: Path | None,
    source_artifact_name: str,
    out_path: Path,
    report_path: Path,
    dry_run: bool,
    printer: Callable[[str], None] = print,
) -> int:
    manifest = load_manifest(manifest_path)
    paths = resolve_paths(
        manifest_path=manifest_path,
        manifest=manifest,
        artifacts_dir=artifacts_dir,
        out_path=out_path,
        report_path=report_path,
    )
    source_artifact = select_source_artifact(manifest, source_artifact_name)
    payload = build_conversion_plan(
        manifest_path=manifest_path,
        manifest=manifest,
        paths=paths,
        source_artifact=source_artifact,
        dry_run=dry_run,
    )

    if dry_run:
        emit_json(payload, printer=printer)
        return 0

    expected_local_source_path = Path(payload["expected_local_source_path"])
    if not expected_local_source_path.is_file():
        raise PreparationError(
            "Expected local source artifact does not exist: "
            f"{expected_local_source_path}. Run Scripts/acquire_silero_artifacts.py first."
        )

    write_report(paths.report_path, payload)
    emit_json(payload, printer=printer)
    return 0


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    try:
        return prepare_conversion(
            manifest_path=args.manifest,
            artifacts_dir=args.artifacts_dir,
            source_artifact_name=args.source_artifact,
            out_path=args.out,
            report_path=args.report,
            dry_run=args.dry_run,
        )
    except PreparationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: filesystem failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
