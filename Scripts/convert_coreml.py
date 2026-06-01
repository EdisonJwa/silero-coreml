#!/usr/bin/env python3
"""Silero VAD JIT to CoreML converter entrypoint.

Dry-run mode remains read-only. Non-dry-run mode verifies the manifest-pinned
JIT artifact and conversion dependencies before tracing and freezing an
explicit-state PyTorch wrapper and saving a CoreML ``.mlpackage``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Sequence, cast

if __package__:
    from .prepare_coreml_conversion import (
        DEFAULT_ARTIFACTS_ROOT,
        DEFAULT_MANIFEST,
        DEFAULT_OUTPUT,
        DEFAULT_REPORT,
        DEFAULT_SOURCE_ARTIFACT,
        TARGET_CONTRACT,
        PreparationError,
        emit_json,
    )
    from .probe_coreml_conversion import build_probe_payload, planned_torchscript_route, write_report
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Scripts.prepare_coreml_conversion import (
        DEFAULT_ARTIFACTS_ROOT,
        DEFAULT_MANIFEST,
        DEFAULT_OUTPUT,
        DEFAULT_REPORT,
        DEFAULT_SOURCE_ARTIFACT,
        TARGET_CONTRACT,
        PreparationError,
        emit_json,
    )
    from Scripts.probe_coreml_conversion import build_probe_payload, planned_torchscript_route, write_report


BLOCKED_EXIT_CODE = 2
ATTEMPTED_ROUTE = "torchscript_explicit_state_conversion"
DIRECT_ONNX_ROUTE = "direct_onnx_conversion"
CONVERSION_DEPENDENCIES = {
    "torch": ("torch",),
    "coremltools": ("coremltools",),
}

TARGET_INPUT = cast(dict[str, object], TARGET_CONTRACT["input"])
TARGET_STATE = cast(dict[str, object], TARGET_CONTRACT["state"])
TARGET_OUTPUTS = cast(dict[str, dict[str, object]], TARGET_CONTRACT["outputs"])
TARGET_INPUT_NAME = str(TARGET_INPUT["name"])
TARGET_INPUT_SHAPE = cast(list[int], TARGET_INPUT["shape"])
TARGET_STATE_NAME = str(TARGET_STATE["name"])
TARGET_STATE_SHAPE = cast(list[int], TARGET_STATE["shape"])
TARGET_OUTPUT_NAME = str(TARGET_OUTPUTS["probability"]["name"])
TARGET_NEXT_STATE_NAME = str(TARGET_OUTPUTS["next_state"]["name"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Silero VAD to CoreML only when a supported route is proven.",
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
        help="Filename of the source artifact to attempt converting.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="CoreML conversion output path. Blocked runs never write this path.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="Path where the JSON conversion report is written in non-dry-run mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the JSON conversion decision without creating files or directories.",
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


def _blocker(component: str, message: str) -> dict[str, str]:
    return {"component": component, "message": message}


class ConversionError(Exception):
    pass


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


def collect_conversion_dependency_checks() -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for module_name, distribution_names in CONVERSION_DEPENDENCIES.items():
        available = _module_available_without_import(module_name)
        payload[module_name] = {
            "available": available,
            "version": _distribution_version(*distribution_names),
            "status": "available" if available else "missing",
            "blocker": None if available else f"Missing conversion dependency: {module_name}",
        }
    return payload


def _conversion_blockers(payload: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []

    source = payload.get("source_artifact_verification", {})
    if isinstance(source, dict):
        if source.get("blocker"):
            blockers.append(_blocker("source_artifact", str(source["blocker"])))
        elif source.get("sha256_status") == "manifest_sha_missing":
            blockers.append(
                _blocker(
                    "source_artifact",
                    "Manifest does not include a SHA-256 for the selected source artifact; "
                    "refusing conversion without source hash verification.",
                )
            )

    selected_source = payload.get("selected_source_artifact", {})
    source_filename = ""
    if isinstance(selected_source, dict):
        source_filename = str(selected_source.get("filename", ""))
    if not source_filename.endswith(".jit"):
        blockers.append(
            _blocker(
                "source_artifact",
                "TorchScript explicit-state conversion requires a manifest-pinned .jit source artifact; "
                f"selected '{source_filename}'.",
            )
        )

    dependencies = payload.get("conversion_dependency_checks", {})
    if isinstance(dependencies, dict):
        for name, details in dependencies.items():
            if isinstance(details, dict) and details.get("blocker"):
                blockers.append(_blocker(f"dependency:{name}", str(details["blocker"])))

    return blockers


def convert_torchscript_to_coreml(*, source_path: Path, out_path: Path) -> dict[str, Any]:
    try:
        torch_module = importlib.import_module("torch")
    except Exception as exc:
        raise ConversionError(f"Failed to import conversion dependency torch: {exc}") from exc

    try:
        coremltools_module = importlib.import_module("coremltools")
    except Exception as exc:
        raise ConversionError(f"Failed to import conversion dependency coremltools: {exc}") from exc

    try:
        torch_module.set_num_threads(1)
    except Exception:
        pass

    class ExplicitStateSileroVAD(torch_module.nn.Module):
        def __init__(self, jit_model):
            super().__init__()
            self.stft = jit_model._model.stft
            self.encoder = jit_model._model.encoder
            self.decoder_head = jit_model._model.decoder.decoder
            rnn = jit_model._model.decoder.rnn
            self._register_lstm_weight("weight_ih", rnn.weight_ih)
            self._register_lstm_weight("weight_hh", rnn.weight_hh)
            self._register_lstm_weight("bias_ih", rnn.bias_ih)
            self._register_lstm_weight("bias_hh", rnn.bias_hh)

        def _register_lstm_weight(self, name, value):
            weight = value.detach().clone()
            if hasattr(self, "register_buffer"):
                self.register_buffer(name, weight)
            else:
                setattr(self, name, weight)

        def forward(self, input_tensor, state_tensor):
            features = self.stft(input_tensor)
            encoded = self.encoder(features).squeeze(-1)
            previous_hidden = state_tensor[0]
            previous_cell = state_tensor[1]
            input_gates = torch_module.nn.functional.linear(
                encoded,
                self.weight_ih,
                self.bias_ih,
            )
            hidden_gates = torch_module.nn.functional.linear(
                previous_hidden,
                self.weight_hh,
                self.bias_hh,
            )
            input_gate, forget_gate, cell_gate, output_gate = (input_gates + hidden_gates).chunk(4, 1)
            next_cell = (
                torch_module.sigmoid(forget_gate) * previous_cell
                + torch_module.sigmoid(input_gate) * torch_module.tanh(cell_gate)
            )
            next_hidden = torch_module.sigmoid(output_gate) * torch_module.tanh(next_cell)
            next_state = torch_module.stack([next_hidden, next_cell])
            output = self.decoder_head(next_hidden.unsqueeze(-1).float())
            return output, next_state

    try:
        jit_model = torch_module.jit.load(str(source_path), map_location=torch_module.device("cpu"))
        jit_model.eval()
        wrapper = ExplicitStateSileroVAD(jit_model).eval()
        example_input = torch_module.zeros(tuple(TARGET_INPUT_SHAPE), dtype=torch_module.float32)
        example_state = torch_module.zeros(tuple(TARGET_STATE_SHAPE), dtype=torch_module.float32)
        with torch_module.no_grad():
            traced_model = torch_module.jit.trace(wrapper, (example_input, example_state), strict=True)
            converted_source_model = torch_module.jit.freeze(traced_model)
    except AttributeError as exc:
        raise ConversionError(
            "Loaded JIT artifact does not expose the expected Silero _model.stft/_model.encoder/_model.decoder modules."
        ) from exc
    except Exception as exc:
        raise ConversionError(f"TorchScript tracing failed: {exc}") from exc

    try:
        tensor_type = coremltools_module.TensorType
        converted_model = coremltools_module.convert(
            converted_source_model,
            convert_to="mlprogram",
            inputs=[
                tensor_type(name=TARGET_INPUT_NAME, shape=tuple(TARGET_INPUT_SHAPE)),
                tensor_type(name=TARGET_STATE_NAME, shape=tuple(TARGET_STATE_SHAPE)),
            ],
            outputs=[
                tensor_type(name=TARGET_OUTPUT_NAME),
                tensor_type(name=TARGET_NEXT_STATE_NAME),
            ],
            minimum_deployment_target=coremltools_module.target.iOS15,
            compute_units=coremltools_module.ComputeUnit.CPU_ONLY,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        converted_model.save(str(out_path))
    except Exception as exc:
        raise ConversionError(f"CoreML conversion failed: {exc}") from exc

    return {
        "status": "converted",
        "source_path": str(source_path),
        "output_path": str(out_path),
        "trace": "torch.jit.trace+torch.jit.freeze",
        "coremltools_convert_to": "mlprogram",
        "minimum_deployment_target": "iOS15",
        "compute_units": "CPU_ONLY",
    }


def build_conversion_decision(
    *,
    manifest_path: Path,
    artifacts_dir: Path | None,
    source_artifact_name: str,
    out_path: Path,
    report_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    resolved_report_path = report_path.resolve()
    payload = build_probe_payload(
        manifest_path=manifest_path,
        artifacts_dir=artifacts_dir,
        source_artifact_name=source_artifact_name,
        out_path=out_path,
        report_path=resolved_report_path,
    )
    blockers = list(payload.get("blockers", []))
    attempted_route = planned_torchscript_route()
    route_status = str(attempted_route["status"])
    payload["conversion_dependency_checks"] = collect_conversion_dependency_checks()
    conversion_blockers = _conversion_blockers(payload)

    payload["dry_run"] = dry_run
    payload["converter_entrypoint"] = "Scripts/convert_coreml.py"
    payload[ATTEMPTED_ROUTE] = attempted_route
    payload["attempted_route"] = {
        "name": ATTEMPTED_ROUTE,
        "status": route_status,
        "supported": not conversion_blockers,
        "preserves_direct_onnx_evidence": DIRECT_ONNX_ROUTE in payload,
    }
    payload["probe_blockers"] = blockers
    payload["blockers"] = conversion_blockers
    payload["overall_status"] = "blocked" if conversion_blockers else "ready_for_conversion"
    payload["conversion_status"] = "blocked_prerequisites" if conversion_blockers else "ready_for_conversion"
    payload["output_written"] = False
    payload["writes_report"] = not dry_run
    payload["report_path"] = str(resolved_report_path)
    return payload


def convert_coreml(
    *,
    manifest_path: Path,
    artifacts_dir: Path | None,
    source_artifact_name: str,
    out_path: Path,
    report_path: Path,
    dry_run: bool,
    printer: Callable[[str], None] = print,
    error_printer: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
) -> int:
    try:
        resolved_report_path = report_path.resolve()
        resolved_out_path = out_path.resolve()
        payload = build_conversion_decision(
            manifest_path=manifest_path,
            artifacts_dir=artifacts_dir,
            source_artifact_name=source_artifact_name,
            out_path=resolved_out_path,
            report_path=resolved_report_path,
            dry_run=dry_run,
        )
        if dry_run:
            emit_json(payload, printer=printer)
            return 0

        if payload["blockers"]:
            emit_json(payload, printer=printer)
            write_report(resolved_report_path, payload)
            error_printer(
                "blocked: CoreML conversion prerequisites are not satisfied; "
                f"wrote report to {resolved_report_path} and did not write {resolved_out_path}"
            )
            return BLOCKED_EXIT_CODE

        source_path = Path(str(payload["expected_local_source_path"]))
        try:
            conversion_result = convert_torchscript_to_coreml(
                source_path=source_path,
                out_path=resolved_out_path,
            )
        except ConversionError as exc:
            payload["overall_status"] = "failed"
            payload["conversion_status"] = "failed"
            payload["conversion_error"] = str(exc)
            payload["output_written"] = False
            emit_json(payload, printer=printer)
            write_report(resolved_report_path, payload)
            error_printer(f"error: {exc}")
            return 1

        payload[ATTEMPTED_ROUTE] = {
            **payload[ATTEMPTED_ROUTE],
            **conversion_result,
        }
        payload["attempted_route"] = {
            **payload["attempted_route"],
            "status": "converted",
            "supported": True,
        }
        payload["overall_status"] = "converted"
        payload["conversion_status"] = "converted"
        payload["output_written"] = True
        payload["output_path"] = str(resolved_out_path)
        emit_json(payload, printer=printer)

        write_report(resolved_report_path, payload)
        return 0
    except PreparationError as exc:
        error_printer(f"error: {exc}")
        return 1
    except OSError as exc:
        error_printer(f"error: filesystem failure: {exc}")
        return 1
    except ConversionError as exc:
        error_printer(f"error: {exc}")
        return 1


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    return convert_coreml(
        manifest_path=args.manifest,
        artifacts_dir=args.artifacts_dir,
        source_artifact_name=args.source_artifact,
        out_path=args.out,
        report_path=args.report,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
