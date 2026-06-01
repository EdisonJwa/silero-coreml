#!/usr/bin/env python3
"""Validate a CoreML Silero VAD export against the official PyTorch model.

This repository intentionally does not vendor any `.mlmodel`, `.mlpackage`,
`.mlmodelc`, or PyTorch weights. The validator therefore requires callers to
pass a model path explicitly:

    python Scripts/validate_coreml.py --model-path /path/to/SileroVAD.mlpackage

The script compares a user-supplied CoreML model against the public
`silero_vad.load_silero_vad()` reference implementation using deterministic
random 16 kHz audio chunks of 512 samples each. It checks three behaviors:

1. Cold-start parity from an all-zero state.
2. Streaming parity while carrying state across consecutive chunks.
3. Reset semantics by replaying the first chunk from a fresh state.

Expected CoreML interface:
  - inputs:  `input` [1, 576] float32, `state_in` [2, 1, 128] float32
  - outputs: `output` scalar-like, `stateN` [2, 1, 128] float32

If Python dependencies are missing, the error message points callers at
`Scripts/requirements.txt` rather than attempting to install anything.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


SAMPLE_RATE = 16_000
CHUNK_SIZE = 512
CONTEXT_SIZE = 64
MODEL_INPUT_SIZE = CONTEXT_SIZE + CHUNK_SIZE
STATE_SHAPE = (2, 1, 128)
INPUT_NAME = "input"
STATE_INPUT_NAME = "state_in"
OUTPUT_NAME = "output"
STATE_OUTPUT_NAME = "stateN"
ALLOWED_MODEL_SUFFIXES = {".mlmodel", ".mlpackage", ".mlmodelc"}


class ValidationError(Exception):
    pass


@dataclass(frozen=True)
class TrialSummary:
    index: int
    cold_start_diff: float
    streaming_max_diff: float
    reset_reference_diff: float
    reset_coreml_replay_diff: float
    reset_pytorch_replay_diff: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a CoreML Silero VAD export against official "
            "silero_vad.load_silero_vad() behavior."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a .mlmodel, .mlpackage, or .mlmodelc Silero VAD export.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of independent random streaming trials to run.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="Number of consecutive 512-sample chunks per trial.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed absolute probability difference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed used for random audio generation.",
    )
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        parser.error(f"model path does not exist: {model_path}")
    if model_path.suffix not in ALLOWED_MODEL_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_MODEL_SUFFIXES))
        parser.error(
            f"model path must end with one of {{{allowed}}}, got: {model_path.name}"
        )
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2 to exercise streaming")
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")

    args.model_path = model_path
    return args


def load_runtime_dependencies():
    missing: list[str] = []
    np_module = None
    ct_module = None
    torch_module = None
    load_silero_vad: Callable[[], object] | None = None

    try:
        np_module = importlib.import_module("numpy")
    except ImportError:
        missing.append("numpy")

    try:
        ct_module = importlib.import_module("coremltools")
    except ImportError:
        missing.append("coremltools")

    try:
        torch_module = importlib.import_module("torch")
    except ImportError:
        missing.append("torch")

    try:
        silero_vad_module = importlib.import_module("silero_vad")
    except ImportError:
        missing.append("silero-vad")
    else:
        candidate = getattr(silero_vad_module, "load_silero_vad", None)
        if callable(candidate):
            load_silero_vad = candidate
        else:
            missing.append("silero-vad")

    if missing:
        packages = ", ".join(missing)
        raise ValidationError(
            "Missing Python dependencies: "
            f"{packages}. Install the validator dependencies referenced by "
            "Scripts/requirements.txt, then retry."
        )

    assert np_module is not None
    assert ct_module is not None
    assert torch_module is not None
    assert load_silero_vad is not None

    torch_module.set_num_threads(1)
    return np_module, ct_module, torch_module, load_silero_vad


def load_reference_model(load_silero_vad, torch_module):
    try:
        model = load_silero_vad()
    except Exception as exc:
        raise ValidationError(
            "Failed to load official silero_vad reference model. "
            "Verify the Python environment described by Scripts/requirements.txt. "
            f"Original error: {exc}"
        ) from exc

    if hasattr(model, "eval"):
        model.eval()
    if not hasattr(model, "reset_states"):
        raise ValidationError(
            "Loaded silero_vad reference model does not expose reset_states(), "
            "so streaming parity cannot be validated."
        )

    return model


def load_coreml_model(coremltools_module, model_path: Path):
    try:
        return coremltools_module.models.MLModel(str(model_path))
    except Exception as exc:
        raise ValidationError(
            f"Failed to load CoreML model at '{model_path}': {exc}"
        ) from exc


def validate_coreml_signature(coreml_model) -> None:
    required_inputs = {INPUT_NAME, STATE_INPUT_NAME}
    required_outputs = {OUTPUT_NAME, STATE_OUTPUT_NAME}

    try:
        spec = coreml_model.get_spec()
    except Exception:
        spec = None

    if spec is None:
        return

    input_features = {feature.name: feature for feature in spec.description.input}
    output_features = {feature.name: feature for feature in spec.description.output}

    missing_inputs = sorted(required_inputs - input_features.keys())
    missing_outputs = sorted(required_outputs - output_features.keys())
    if missing_inputs or missing_outputs:
        problems = []
        if missing_inputs:
            problems.append(f"missing inputs: {', '.join(missing_inputs)}")
        if missing_outputs:
            problems.append(f"missing outputs: {', '.join(missing_outputs)}")
        raise ValidationError(
            "CoreML model does not expose the expected explicit-state interface ("
            + "; ".join(problems)
            + ")."
        )

    audio_shape = feature_shape(input_features[INPUT_NAME])
    if audio_shape is not None and audio_shape != (1, MODEL_INPUT_SIZE):
        raise ValidationError(
            f"CoreML input '{INPUT_NAME}' must have shape [1, {MODEL_INPUT_SIZE}], "
            f"got {list(audio_shape)}"
        )

    state_in_shape = feature_shape(input_features[STATE_INPUT_NAME])
    if state_in_shape is not None and state_in_shape != STATE_SHAPE:
        raise ValidationError(
            f"CoreML input '{STATE_INPUT_NAME}' must have shape [2, 1, 128], "
            f"got {list(state_in_shape)}"
        )

    state_out_shape = feature_shape(output_features[STATE_OUTPUT_NAME])
    if state_out_shape is not None and state_out_shape != STATE_SHAPE:
        raise ValidationError(
            f"CoreML output '{STATE_OUTPUT_NAME}' must have shape [2, 1, 128], "
            f"got {list(state_out_shape)}"
        )


def feature_shape(feature) -> tuple[int, ...] | None:
    try:
        multi_array_type = feature.type.multiArrayType
    except AttributeError:
        return None

    shape = tuple(int(dimension) for dimension in multi_array_type.shape)
    return shape or None


def scalarize_probability(np_module, value, *, output_name: str) -> float:
    array = np_module.asarray(value, dtype=np_module.float32)
    if array.size != 1:
        raise ValidationError(
            f"CoreML output '{output_name}' must be scalar-like, got shape {list(array.shape)}"
        )
    return float(array.reshape(-1)[0])


def zero_state(np_module):
    return np_module.zeros(STATE_SHAPE, dtype=np_module.float32)


def zero_context(np_module):
    return np_module.zeros(CONTEXT_SIZE, dtype=np_module.float32)


def generate_random_stream(np_module, sequence_length: int, seed: int, trial_index: int):
    rng = np_module.random.default_rng(seed + trial_index)
    return rng.uniform(
        low=-1.0,
        high=1.0,
        size=(sequence_length, CHUNK_SIZE),
    ).astype(np_module.float32)


def predict_pytorch(torch_model, torch_module, chunk) -> float:
    tensor = torch_module.from_numpy(chunk.copy())
    with torch_module.no_grad():
        output = torch_model(tensor, SAMPLE_RATE)
    return float(output.detach().cpu().reshape(-1)[0].item())


def predict_coreml(coreml_model, np_module, chunk, state_in, context):
    chunk_array = np_module.asarray(chunk, dtype=np_module.float32).reshape(CHUNK_SIZE)
    context_array = np_module.asarray(context, dtype=np_module.float32).reshape(CONTEXT_SIZE)
    model_input = np_module.concatenate([context_array, chunk_array], axis=0).reshape(
        1,
        MODEL_INPUT_SIZE,
    )
    inputs = {
        INPUT_NAME: model_input,
        STATE_INPUT_NAME: np_module.asarray(state_in, dtype=np_module.float32).reshape(STATE_SHAPE),
    }

    try:
        outputs = coreml_model.predict(inputs)
    except Exception as exc:
        raise ValidationError(
            "CoreML prediction failed. Confirm the model uses the explicit-state "
            f"interface `{INPUT_NAME}` + `{STATE_INPUT_NAME}` -> "
            f"`{OUTPUT_NAME}` + `{STATE_OUTPUT_NAME}`. "
            f"Original error: {exc}"
        ) from exc

    if OUTPUT_NAME not in outputs or STATE_OUTPUT_NAME not in outputs:
        raise ValidationError(
            f"CoreML prediction outputs must include '{OUTPUT_NAME}' and '{STATE_OUTPUT_NAME}'. "
            f"Received outputs: {sorted(outputs.keys())}"
        )

    probability = scalarize_probability(
        np_module,
        outputs[OUTPUT_NAME],
        output_name=OUTPUT_NAME,
    )
    state_out = np_module.asarray(outputs[STATE_OUTPUT_NAME], dtype=np_module.float32)
    if state_out.shape != STATE_SHAPE:
        raise ValidationError(
            f"CoreML output '{STATE_OUTPUT_NAME}' must have shape [2, 1, 128], "
            f"got {list(state_out.shape)}"
        )
    next_context = chunk_array[-CONTEXT_SIZE:]
    return probability, state_out, next_context


def run_trial(
    *,
    trial_index: int,
    sequence,
    np_module,
    torch_module,
    torch_model,
    coreml_model,
) -> TrialSummary:
    torch_model.reset_states()
    state = zero_state(np_module)
    context = zero_context(np_module)

    first_chunk = sequence[0]
    cold_start_reference = predict_pytorch(torch_model, torch_module, first_chunk)
    cold_start_coreml, state, context = predict_coreml(
        coreml_model,
        np_module,
        first_chunk,
        state,
        context,
    )
    cold_start_diff = abs(cold_start_reference - cold_start_coreml)

    streaming_max_diff = cold_start_diff
    for chunk in sequence[1:]:
        reference_probability = predict_pytorch(torch_model, torch_module, chunk)
        coreml_probability, state, context = predict_coreml(
            coreml_model,
            np_module,
            chunk,
            state,
            context,
        )
        streaming_max_diff = max(
            streaming_max_diff,
            abs(reference_probability - coreml_probability),
        )

    torch_model.reset_states()
    reset_reference = predict_pytorch(torch_model, torch_module, first_chunk)
    reset_coreml, _, _ = predict_coreml(
        coreml_model,
        np_module,
        first_chunk,
        zero_state(np_module),
        zero_context(np_module),
    )

    return TrialSummary(
        index=trial_index,
        cold_start_diff=cold_start_diff,
        streaming_max_diff=streaming_max_diff,
        reset_reference_diff=abs(reset_reference - reset_coreml),
        reset_coreml_replay_diff=abs(reset_coreml - cold_start_coreml),
        reset_pytorch_replay_diff=abs(reset_reference - cold_start_reference),
    )


def print_trial_summary(summary: TrialSummary) -> None:
    print(
        f"trial={summary.index:02d} "
        f"cold_start_diff={summary.cold_start_diff:.8f} "
        f"streaming_max_diff={summary.streaming_max_diff:.8f} "
        f"reset_reference_diff={summary.reset_reference_diff:.8f} "
        f"reset_coreml_replay_diff={summary.reset_coreml_replay_diff:.8f} "
        f"reset_pytorch_replay_diff={summary.reset_pytorch_replay_diff:.8f}"
    )


def max_metric(summaries: Iterable[TrialSummary]) -> float:
    maximum = 0.0
    for summary in summaries:
        maximum = max(
            maximum,
            summary.cold_start_diff,
            summary.streaming_max_diff,
            summary.reset_reference_diff,
            summary.reset_coreml_replay_diff,
            summary.reset_pytorch_replay_diff,
        )
    return maximum


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    try:
        np_module, ct_module, torch_module, load_silero_vad = load_runtime_dependencies()
        torch_model = load_reference_model(load_silero_vad, torch_module)
        coreml_model = load_coreml_model(ct_module, args.model_path)
        validate_coreml_signature(coreml_model)

        summaries: list[TrialSummary] = []
        for trial_index in range(1, args.trials + 1):
            sequence = generate_random_stream(
                np_module=np_module,
                sequence_length=args.sequence_length,
                seed=args.seed,
                trial_index=trial_index,
            )
            summary = run_trial(
                trial_index=trial_index,
                sequence=sequence,
                np_module=np_module,
                torch_module=torch_module,
                torch_model=torch_model,
                coreml_model=coreml_model,
            )
            summaries.append(summary)
            print_trial_summary(summary)

        overall_max_diff = max_metric(summaries)
        print(
            f"overall_max_diff={overall_max_diff:.8f} "
            f"tolerance={args.tolerance:.8f}"
        )

        if overall_max_diff > args.tolerance:
            print(
                "Validation failed: CoreML model diverged from the official "
                "PyTorch Silero VAD beyond tolerance.",
                file=sys.stderr,
            )
            return 1

        print("Validation passed: CoreML model matches reference behavior within tolerance.")
        return 0
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
