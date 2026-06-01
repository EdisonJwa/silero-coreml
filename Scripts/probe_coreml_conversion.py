#!/usr/bin/env python3
"""Probe Silero VAD CoreML conversion feasibility without converting assets.

The probe is intentionally read-only by default. It verifies manifest/source
artifact provenance, inspects the ONNX interface when local tooling exists,
runs a small ONNX Runtime smoke prediction when possible, and reports the
currently observed CoreMLTools ONNX route status. It does not write model
outputs and only writes a JSON report when ``--report`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence, cast

if __package__:
    from .prepare_coreml_conversion import (
        DEFAULT_ARTIFACTS_ROOT,
        DEFAULT_MANIFEST,
        DEFAULT_OUTPUT,
        DEFAULT_SOURCE_ARTIFACT,
        TARGET_CONTRACT,
        PreparationError,
        build_conversion_plan,
        emit_json,
        load_manifest,
        resolve_paths,
        select_source_artifact,
    )
    from .report_coreml_conversion_readiness import summarize_onnx_model
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Scripts.prepare_coreml_conversion import (
        DEFAULT_ARTIFACTS_ROOT,
        DEFAULT_MANIFEST,
        DEFAULT_OUTPUT,
        DEFAULT_SOURCE_ARTIFACT,
        TARGET_CONTRACT,
        PreparationError,
        build_conversion_plan,
        emit_json,
        load_manifest,
        resolve_paths,
        select_source_artifact,
    )
    from Scripts.report_coreml_conversion_readiness import summarize_onnx_model


TARGET_INPUT = cast(dict[str, object], TARGET_CONTRACT["input"])
TARGET_STATE = cast(dict[str, object], TARGET_CONTRACT["state"])
TARGET_OUTPUTS = cast(dict[str, dict[str, object]], TARGET_CONTRACT["outputs"])
TARGET_PROBABILITY_OUTPUT = TARGET_OUTPUTS["probability"]
TARGET_NEXT_STATE_OUTPUT = TARGET_OUTPUTS["next_state"]
TARGET_INPUT_NAME = str(TARGET_INPUT["name"])
TARGET_INPUT_SHAPE = cast(list[int], TARGET_INPUT["shape"])
TARGET_STATE_NAME = str(TARGET_STATE["name"])
TARGET_STATE_SHAPE = cast(list[int], TARGET_STATE["shape"])
TARGET_PROBABILITY_OUTPUT_NAME = str(TARGET_PROBABILITY_OUTPUT["name"])
TARGET_NEXT_STATE_OUTPUT_NAME = str(TARGET_NEXT_STATE_OUTPUT["name"])
TARGET_NEXT_STATE_OUTPUT_SHAPE = cast(list[int], TARGET_NEXT_STATE_OUTPUT["shape"])

REQUIRED_DEPENDENCIES = {
    "onnx": ("onnx",),
    "onnxruntime": ("onnxruntime",),
    "coremltools": ("coremltools",),
    "torch": ("torch",),
    "numpy": ("numpy",),
}

COREMLTOOLS_9_RELEASE_NOTE_FACTS = [
    "CoreMLTools 9.0 release notes mention Python 3.13 support.",
    "CoreMLTools 9.0 release notes mention PyTorch 2.7 support.",
    "CoreMLTools 9.0 release notes mention state read/write support.",
    "CoreMLTools 9.0 release notes do not explicitly mention ONNX conversion support.",
]

COREMLTOOLS_FALLBACK_RECOMMENDATION = (
    "Do not plan on direct ONNX conversion with CoreMLTools 9.0. Rebuild or "
    "export an explicit-state PyTorch module that exposes input,state_in -> "
    "output,stateN, trace and freeze it with torch.jit.trace and torch.jit.freeze, "
    "convert that frozen module with CoreMLTools, then validate CoreML parity before packaging."
)
TORCHSCRIPT_ROUTE_NAME = "torchscript_explicit_state_conversion"
TORCHSCRIPT_ROUTE_RECOMMENDATION = (
    "Build an explicit-state PyTorch wrapper around the official pretrained Silero model, "
    "trace and freeze it with torch.jit.trace and torch.jit.freeze using input [1,576] and state_in [2,1,128], convert "
    "the frozen TorchScript module with CoreMLTools, then run parity validation before packaging."
)


def planned_torchscript_route() -> dict[str, Any]:
    return {
        "status": "implemented",
        "supported": True,
        "source": "explicit-state PyTorch wrapper traced and frozen with torch.jit.trace and torch.jit.freeze",
        "target_contract": "input,state_in -> output,stateN",
        "apple_reference": "https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html",
        "recommendation": TORCHSCRIPT_ROUTE_RECOMMENDATION,
        "blocker": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe manifest-driven Silero VAD CoreML conversion feasibility.",
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
        help="Filename of the source artifact to probe for conversion feasibility.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Planned CoreML conversion output path. The probe never writes this path.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path for writing the JSON feasibility report. Omit to only print stdout.",
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
    args.report = Path(args.report).expanduser().resolve() if args.report is not None else None

    if not isinstance(args.source_artifact, str) or not args.source_artifact:
        parser.error("source artifact must be a non-empty filename")
    if Path(args.source_artifact).name != args.source_artifact:
        parser.error(f"source artifact must not contain path separators: {args.source_artifact}")

    return args


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


def collect_dependency_checks() -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for module_name, distribution_names in REQUIRED_DEPENDENCIES.items():
        available = _module_available_without_import(module_name)
        payload[module_name] = {
            "available": available,
            "version": _distribution_version(*distribution_names),
            "status": "available" if available else "missing",
            "blocker": None if available else f"Missing optional probe dependency: {module_name}",
        }
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_artifact(source_path: Path, expected_sha256: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "available" if source_path.is_file() else "missing_artifact",
        "path": str(source_path),
        "exists": source_path.is_file(),
        "expected_sha256": expected_sha256,
        "actual_sha256": None,
        "sha256_status": "not_checked",
        "blocker": None,
    }

    if not source_path.is_file():
        payload["blocker"] = (
            f"Expected local source artifact does not exist: {source_path}. "
            "Run Scripts/acquire_silero_artifacts.py first."
        )
        return payload

    actual_sha256 = sha256_file(source_path)
    payload["actual_sha256"] = actual_sha256

    if expected_sha256 is None:
        payload["sha256_status"] = "manifest_sha_missing"
        return payload

    if actual_sha256 == expected_sha256:
        payload["sha256_status"] = "match"
        return payload

    payload["status"] = "sha256_mismatch"
    payload["sha256_status"] = "mismatch"
    payload["blocker"] = "Local source artifact SHA-256 does not match the manifest."
    return payload


def skip_unverified_onnx_probe(source_path: Path, verification: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "skipped_unverified_artifact",
        "source_path": str(source_path),
        "required_sha256_status": "match",
        "actual_sha256_status": verification.get("sha256_status"),
        "blocker": "Skipping ONNX parsing/runtime smoke because source artifact SHA-256 is not verified.",
    }


def skip_non_onnx_probe(source_path: Path) -> dict[str, Any]:
    return {
        "status": "skipped_non_onnx_source",
        "source_path": str(source_path),
        "blocker": None,
    }


def inspect_onnx_interface(source_path: Path) -> dict[str, Any]:
    if not source_path.is_file():
        return {"status": "skipped_missing_artifact", "source_path": str(source_path)}

    try:
        onnx_module = importlib.import_module("onnx")
    except ModuleNotFoundError:
        return {
            "status": "skipped_missing_dependency",
            "source_path": str(source_path),
            "dependency": "onnx",
            "blocker": "Cannot inspect ONNX interface because dependency 'onnx' is missing.",
        }

    load_function = getattr(onnx_module, "load_model", None) or getattr(onnx_module, "load", None)
    if not callable(load_function):
        return {
            "status": "failed",
            "source_path": str(source_path),
            "blocker": "onnx is installed but does not expose load_model(...) or load(...).",
        }

    try:
        model = load_function(str(source_path))
        payload = summarize_onnx_model(onnx_module, model)
    except Exception as exc:
        return {
            "status": "failed",
            "source_path": str(source_path),
            "blocker": f"ONNX interface inspection failed: {exc}",
        }

    payload["source_path"] = str(source_path)
    payload["node_count"] = sum(payload.get("node_type_counts", {}).values())
    return payload


def _shape_of_output(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _output_contract_matches(output_shapes: dict[str, list[int] | None]) -> bool:
    return (
        TARGET_PROBABILITY_OUTPUT_NAME in output_shapes
        and TARGET_NEXT_STATE_OUTPUT_NAME in output_shapes
        and output_shapes[TARGET_NEXT_STATE_OUTPUT_NAME] == TARGET_NEXT_STATE_OUTPUT_SHAPE
    )


def run_onnxruntime_smoke(source_path: Path) -> dict[str, Any]:
    if not source_path.is_file():
        return {"status": "skipped_missing_artifact", "source_path": str(source_path)}

    try:
        numpy_module = importlib.import_module("numpy")
    except ModuleNotFoundError:
        return {
            "status": "skipped_missing_dependency",
            "source_path": str(source_path),
            "dependency": "numpy",
            "blocker": "Cannot run ONNX Runtime smoke because dependency 'numpy' is missing.",
        }

    try:
        ort_module = importlib.import_module("onnxruntime")
    except ModuleNotFoundError:
        return {
            "status": "skipped_missing_dependency",
            "source_path": str(source_path),
            "dependency": "onnxruntime",
            "blocker": "Cannot run ONNX Runtime smoke because dependency 'onnxruntime' is missing.",
        }

    try:
        session = ort_module.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
        feed = {
            TARGET_INPUT_NAME: numpy_module.zeros(TARGET_INPUT_SHAPE, dtype=numpy_module.float32),
            TARGET_STATE_NAME: numpy_module.zeros(TARGET_STATE_SHAPE, dtype=numpy_module.float32),
            "sr": numpy_module.array(16000, dtype=numpy_module.int64),
        }
        try:
            output_values = session.run(None, feed)
        except Exception:
            feed["sr"] = numpy_module.array([16000], dtype=numpy_module.int64)
            output_values = session.run(None, feed)

        output_names = [str(getattr(item, "name", "")) for item in session.get_outputs()]
        output_shapes = {
            name: _shape_of_output(value) for name, value in zip(output_names, output_values, strict=False)
        }
        return {
            "status": "passed" if _output_contract_matches(output_shapes) else "output_contract_mismatch",
            "source_path": str(source_path),
            "input_feed_shapes": {
                TARGET_INPUT_NAME: TARGET_INPUT_SHAPE,
                TARGET_STATE_NAME: TARGET_STATE_SHAPE,
                "sr": [],
            },
            "output_names": output_names,
            "output_shapes": output_shapes,
            "expected_output_names": [
                TARGET_PROBABILITY_OUTPUT_NAME,
                TARGET_NEXT_STATE_OUTPUT_NAME,
            ],
            "matches_expected_contract": _output_contract_matches(output_shapes),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "source_path": str(source_path),
            "blocker": f"ONNX Runtime smoke prediction failed: {exc}",
        }


def _callable_has_onnx_hint(function: Any) -> bool:
    text_parts = [str(function)]
    annotations = getattr(function, "__annotations__", None)
    if annotations:
        text_parts.append(str(annotations))
    doc = getattr(function, "__doc__", None)
    if doc:
        text_parts.append(doc)
    return "onnx" in " ".join(text_parts).lower()


def probe_coremltools_route() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "unsupported",
        "dependency": "coremltools",
        "available": False,
        "version": _distribution_version("coremltools"),
        "direct_onnx_symbols": [],
        "warnings": [],
        "stderr": "",
        "release_note_facts": COREMLTOOLS_9_RELEASE_NOTE_FACTS,
        "recommendation": COREMLTOOLS_FALLBACK_RECOMMENDATION,
        "blocker": "CoreMLTools 9.0 does not expose an observed direct ONNX conversion route.",
    }

    if not _module_available_without_import("coremltools"):
        payload["status"] = "skipped_missing_dependency"
        payload["blocker"] = "Cannot probe CoreMLTools conversion route because dependency 'coremltools' is missing."
        return payload

    stderr = io.StringIO()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stderr(stderr):
                coremltools_module = importlib.import_module("coremltools")
        except ModuleNotFoundError:
            payload["status"] = "skipped_missing_dependency"
            payload["stderr"] = stderr.getvalue().strip()
            payload["blocker"] = "Cannot import CoreMLTools even though module discovery found it."
            return payload
        except Exception as exc:
            payload["status"] = "failed"
            payload["stderr"] = stderr.getvalue().strip()
            payload["blocker"] = f"CoreMLTools import failed during route probe: {exc}"
            return payload

    payload["available"] = True
    payload["version"] = str(getattr(coremltools_module, "__version__", payload["version"]))
    payload["warnings"] = [str(item.message) for item in captured_warnings]
    payload["stderr"] = stderr.getvalue().strip()

    direct_symbols: list[str] = []
    converters = getattr(coremltools_module, "converters", None)
    if converters is not None and hasattr(converters, "onnx"):
        direct_symbols.append("coremltools.converters.onnx")
    if hasattr(coremltools_module, "onnx"):
        direct_symbols.append("coremltools.onnx")
    convert_function = getattr(coremltools_module, "convert", None)
    if callable(convert_function) and _callable_has_onnx_hint(convert_function):
        direct_symbols.append("coremltools.convert accepts/mentions ONNX")

    payload["direct_onnx_symbols"] = direct_symbols
    if direct_symbols:
        payload["status"] = "observed"
        payload["blocker"] = None
        payload["recommendation"] = "Inspect observed direct ONNX symbols before relying on them."

    return payload


def _collect_blockers(payload: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    selected_source = payload.get("selected_source_artifact", {})
    source_filename = ""
    if isinstance(selected_source, dict):
        source_filename = str(selected_source.get("filename", ""))
    source_is_onnx = source_filename.endswith(".onnx")

    source = payload.get("source_artifact_verification", {})
    if isinstance(source, dict) and source.get("blocker"):
        blockers.append({"component": "source_artifact", "message": str(source["blocker"])})

    dependencies = payload.get("dependency_checks", {})
    if isinstance(dependencies, dict):
        for name, details in dependencies.items():
            if not source_is_onnx and name in {"onnx", "onnxruntime", "numpy"}:
                continue
            if isinstance(details, dict) and details.get("blocker"):
                blockers.append({"component": f"dependency:{name}", "message": str(details["blocker"])})

    for key, component in (
        ("onnx_interface", "onnx_interface"),
        ("onnxruntime_smoke", "onnxruntime_smoke"),
        ("direct_onnx_conversion", "direct_onnx_conversion"),
        (TORCHSCRIPT_ROUTE_NAME, TORCHSCRIPT_ROUTE_NAME),
    ):
        if not source_is_onnx and key in {"onnx_interface", "onnxruntime_smoke"}:
            continue
        details = payload.get(key, {})
        if isinstance(details, dict) and details.get("blocker"):
            blockers.append({"component": component, "message": str(details["blocker"])})

    return blockers


def build_probe_payload(
    *,
    manifest_path: Path,
    artifacts_dir: Path | None,
    source_artifact_name: str,
    out_path: Path,
    report_path: Path | None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    paths = resolve_paths(
        manifest_path=manifest_path,
        manifest=manifest,
        artifacts_dir=artifacts_dir,
        out_path=out_path,
        report_path=report_path or out_path.with_suffix(".probe-report.json"),
    )
    source_artifact = select_source_artifact(manifest, source_artifact_name)
    payload = build_conversion_plan(
        manifest_path=manifest_path,
        manifest=manifest,
        paths=paths,
        source_artifact=source_artifact,
        dry_run=True,
    )
    source_path = Path(payload["expected_local_source_path"])

    payload["probe_type"] = "coreml_conversion_feasibility"
    payload["report_path"] = str(report_path) if report_path is not None else None
    payload["writes_report"] = report_path is not None
    source_verification = verify_source_artifact(source_path, source_artifact.sha256)
    payload["source_artifact_verification"] = source_verification
    payload["dependency_checks"] = collect_dependency_checks()
    if source_verification.get("sha256_status") == "match" and source_path.suffix.lower() == ".onnx":
        payload["onnx_interface"] = inspect_onnx_interface(source_path)
        payload["onnxruntime_smoke"] = run_onnxruntime_smoke(source_path)
    elif source_verification.get("sha256_status") == "match":
        payload["onnx_interface"] = skip_non_onnx_probe(source_path)
        payload["onnxruntime_smoke"] = skip_non_onnx_probe(source_path)
    else:
        payload["onnx_interface"] = skip_unverified_onnx_probe(source_path, source_verification)
        payload["onnxruntime_smoke"] = skip_unverified_onnx_probe(source_path, source_verification)
    payload["direct_onnx_conversion"] = probe_coremltools_route()
    payload[TORCHSCRIPT_ROUTE_NAME] = planned_torchscript_route()

    blockers = _collect_blockers(payload)
    payload["blockers"] = blockers
    payload["overall_status"] = "blocked" if blockers else "ready_for_next_probe"
    return payload


def write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def probe_coreml_conversion(
    *,
    manifest_path: Path,
    artifacts_dir: Path | None,
    source_artifact_name: str,
    out_path: Path,
    report_path: Path | None,
    printer: Callable[[str], None] = print,
    error_printer: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
) -> int:
    try:
        payload = build_probe_payload(
            manifest_path=manifest_path,
            artifacts_dir=artifacts_dir,
            source_artifact_name=source_artifact_name,
            out_path=out_path,
            report_path=report_path,
        )
        if report_path is not None:
            write_report(report_path, payload)
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
    return probe_coreml_conversion(
        manifest_path=args.manifest,
        artifacts_dir=args.artifacts_dir,
        source_artifact_name=args.source_artifact,
        out_path=args.out,
        report_path=args.report,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
