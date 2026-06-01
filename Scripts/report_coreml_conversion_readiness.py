#!/usr/bin/env python3
"""Report Silero VAD CoreML conversion readiness without writing files.

This script stays read-only by design. It reuses the manifest-driven conversion
planning helpers from ``prepare_coreml_conversion.py`` and augments the payload
with lightweight dependency probing plus optional ONNX graph introspection when
the selected source is an ONNX artifact and that artifact is already present.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__:
    from .prepare_coreml_conversion import (
        DEFAULT_ARTIFACTS_ROOT,
        DEFAULT_MANIFEST,
        DEFAULT_OUTPUT,
        DEFAULT_REPORT,
        DEFAULT_SOURCE_ARTIFACT,
        PreparationError,
        build_conversion_plan,
        default_artifacts_dir,
        emit_json,
        load_manifest,
        resolve_paths,
        select_source_artifact,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Scripts.prepare_coreml_conversion import (
        DEFAULT_ARTIFACTS_ROOT,
        DEFAULT_MANIFEST,
        DEFAULT_OUTPUT,
        DEFAULT_REPORT,
        DEFAULT_SOURCE_ARTIFACT,
        PreparationError,
        build_conversion_plan,
        default_artifacts_dir,
        emit_json,
        load_manifest,
        resolve_paths,
        select_source_artifact,
    )


DEFAULT_READINESS_REPORT = DEFAULT_REPORT.with_name("SileroVAD.readiness-report.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report a manifest-driven Silero VAD CoreML conversion readiness summary.",
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
        help=(
            "Directory containing pinned local upstream artifacts. Defaults to "
            f"{DEFAULT_ARTIFACTS_ROOT}/<manifest upstream commit>."
        ),
    )
    parser.add_argument(
        "--source-artifact",
        default=DEFAULT_SOURCE_ARTIFACT,
        help="Filename of the source artifact to assess for conversion readiness.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Planned CoreML conversion output path.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_READINESS_REPORT),
        help="Planned readiness report path. This command never writes the file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the JSON readiness payload without creating files or directories.",
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


def _default_error_printer(line: str) -> None:
    print(line, file=sys.stderr)


def _module_available_without_import(module_name: str) -> bool:
    if module_name in sys.modules:
        return True

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _distribution_version(*distribution_names: str) -> str | None:
    for distribution_name in distribution_names:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def collect_dependency_availability() -> dict[str, dict[str, Any]]:
    dependencies = {
        "onnx": ("onnx",),
        "onnxruntime": ("onnxruntime",),
        "coremltools": ("coremltools",),
        "numpy": ("numpy",),
        "torch": ("torch",),
        "silero_vad": ("silero-vad", "silero_vad"),
    }

    payload: dict[str, dict[str, Any]] = {}
    for module_name, distribution_names in dependencies.items():
        payload[module_name] = {
            "available": _module_available_without_import(module_name),
            "version": _distribution_version(*distribution_names),
        }
    return payload


def _has_field(message: object, field_name: str) -> bool:
    has_field = getattr(message, "HasField", None)
    if callable(has_field):
        try:
            return bool(has_field(field_name))
        except ValueError:
            return False
    return hasattr(message, field_name)


def _shape_dim_value(dim: object) -> int | str | None:
    if _has_field(dim, "dim_value"):
        return getattr(dim, "dim_value")
    if _has_field(dim, "dim_param"):
        return getattr(dim, "dim_param")
    return None


def _tensor_dtype_name(onnx_module: Any, elem_type: object) -> str | None:
    tensor_proto = getattr(onnx_module, "TensorProto", None)
    data_type = getattr(tensor_proto, "DataType", None)
    name_function = getattr(data_type, "Name", None)
    if callable(name_function):
        try:
            return str(name_function(elem_type))
        except Exception:
            return None
    return None


def _value_info_summary(onnx_module: Any, value_info: object) -> dict[str, Any]:
    summary = {
        "name": str(getattr(value_info, "name", "")),
        "dtype": None,
        "shape": [],
    }

    value_type = getattr(value_info, "type", None)
    if value_type is None or not _has_field(value_type, "tensor_type"):
        return summary

    tensor_type = getattr(value_type, "tensor_type", None)
    if tensor_type is None:
        return summary

    elem_type = getattr(tensor_type, "elem_type", None)
    if elem_type is not None:
        summary["dtype"] = _tensor_dtype_name(onnx_module, elem_type)

    if _has_field(tensor_type, "shape"):
        shape = getattr(tensor_type, "shape", None)
        dims = getattr(shape, "dim", []) if shape is not None else []
        summary["shape"] = [_shape_dim_value(dim) for dim in dims]

    return summary


def summarize_onnx_model(onnx_module: Any, model: Any) -> dict[str, Any]:
    graph = getattr(model, "graph", None)
    if graph is None:
        raise PreparationError("ONNX model is missing a graph payload.")

    node_type_counts = Counter(str(getattr(node, "op_type", "")) for node in getattr(graph, "node", []))

    return {
        "status": "available",
        "inputs": [_value_info_summary(onnx_module, item) for item in getattr(graph, "input", [])],
        "outputs": [_value_info_summary(onnx_module, item) for item in getattr(graph, "output", [])],
        "opset_imports": [
            {
                "domain": str(getattr(opset_import, "domain", "")),
                "version": int(getattr(opset_import, "version", 0)),
            }
            for opset_import in getattr(model, "opset_import", [])
        ],
        "initializer_count": len(getattr(graph, "initializer", [])),
        "node_type_counts": dict(sorted(node_type_counts.items())),
    }


def inspect_onnx_readiness(expected_local_source_path: Path) -> dict[str, Any]:
    if expected_local_source_path.suffix.lower() != ".onnx":
        return {
            "status": "skipped_non_onnx_source",
            "source_path": str(expected_local_source_path),
        }

    if not expected_local_source_path.is_file():
        return {
            "status": "skipped_missing_artifact",
            "source_path": str(expected_local_source_path),
        }

    try:
        onnx_module = importlib.import_module("onnx")
    except ModuleNotFoundError:
        return {
            "status": "skipped_missing_dependency",
            "source_path": str(expected_local_source_path),
            "dependency": "onnx",
        }

    load_function = getattr(onnx_module, "load_model", None) or getattr(onnx_module, "load", None)
    if not callable(load_function):
        raise PreparationError("onnx is installed but does not expose load_model(...) or load(...).")

    model = load_function(str(expected_local_source_path))
    payload = summarize_onnx_model(onnx_module, model)
    payload["source_path"] = str(expected_local_source_path)
    return payload


def report_readiness(
    *,
    manifest_path: Path,
    artifacts_dir: Path | None,
    source_artifact_name: str,
    out_path: Path,
    report_path: Path,
    dry_run: bool,
    printer: Callable[[str], None] = print,
    error_printer: Callable[[str], None] = _default_error_printer,
) -> int:
    try:
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
        payload["dependency_availability"] = collect_dependency_availability()
        payload["onnx_introspection"] = inspect_onnx_readiness(
            Path(payload["expected_local_source_path"])
        )
        emit_json(payload, printer=printer)
        return 0
    except PreparationError as exc:
        error_printer(f"error: {exc}")
        return 1
    except OSError as exc:
        error_printer(f"error: filesystem failure: {exc}")
        return 1


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    return report_readiness(
        manifest_path=args.manifest,
        artifacts_dir=args.artifacts_dir,
        source_artifact_name=args.source_artifact,
        out_path=args.out,
        report_path=args.report,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
