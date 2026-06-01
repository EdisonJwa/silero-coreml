#!/usr/bin/env python3
"""Safely validate a local CoreML Silero VAD model with a WAV or microphone clip.

The CLI is intentionally dependency-light at parse/help time. CoreML, numpy, and
coremltools are imported only when audio analysis actually runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, cast


SAMPLE_RATE = 16_000
HOP_SIZE = 512
DEFAULT_THRESHOLD = 0.5
DEFAULT_MODEL_PATH = Path("Sources/SileroCoreML/Resources/SileroVAD.mlpackage")
SUPPORTED_RECORDERS = ("sox", "rec", "ffmpeg", "afrecord")


class MicrophoneValidationError(Exception):
    pass


@dataclass(frozen=True)
class ParsedArgs:
    input_wav: Path | None
    duration: float | None
    output: Path | None
    model_path: Path
    threshold: float
    json_output: Path | None
    dry_run: bool
    repo_root: Path


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float
    duration: float
    frames: int


@dataclass(frozen=True)
class VadSummary:
    source: str
    threshold: float
    frames: int
    audio_seconds: float
    speech_frames: int
    speech_seconds: float
    speech_ratio: float
    segments: list[SpeechSegment]
    probability: dict[str, float]
    timing: dict[str, float | int]


class Recorder(Protocol):
    name: str

    def record(self, *, duration: float, output: Path) -> None: ...


class ParsedArgsNamespace(Protocol):
    input_wav: str | None
    duration: float | None
    output: str | None
    model_path: str
    threshold: float
    json_output: str | None
    dry_run: bool


class ValidateCoreMLModuleLike(Protocol):
    def load_coreml_model(self, coremltools_module: object, model_path: Path) -> object: ...

    def validate_coreml_signature(self, coreml_model: object) -> None: ...

    def zero_state(self, np_module: object) -> object: ...

    def zero_context(self, np_module: object) -> object: ...

    def predict_coreml(
        self,
        coreml_model: object,
        np_module: object,
        chunk: Sequence[float],
        state_in: object,
        context: object,
    ) -> tuple[float, object, object]: ...


class EvaluateTenVadModuleLike(Protocol):
    def load_pcm_wave_mono_16khz_stdlib(self, path: Path) -> list[float]: ...

    def time_call(
        self,
        callback: Callable[[], tuple[float, object, object]],
        *,
        clock: Callable[[], float],
        latencies: list[float],
    ) -> tuple[float, object, object]: ...

    def summarize_latency(self, latencies_seconds: Sequence[float]) -> dict[str, float | int]: ...

    def chunk_audio_frames(self, samples: Sequence[float], *, chunk_size: int) -> list[list[float]]: ...


class RecorderCommand:
    name: str
    executable: str

    def __init__(self, name: str, executable: str) -> None:
        self.name = name
        self.executable = executable

    def record(self, *, duration: float, output: Path) -> None:
        command = recorder_command_args(self.name, self.executable, duration, output)
        try:
            _ = subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise MicrophoneValidationError(missing_recorder_message()) from exc
        except subprocess.CalledProcessError as exc:
            raise MicrophoneValidationError(recording_failed_message(self.name, exc)) from exc


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate CoreML Silero VAD probabilities from an existing 16 kHz PCM WAV "
            "or an explicit short microphone recording."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--input-wav", help="Existing uncompressed 16 kHz PCM WAV to analyze.")
    _ = parser.add_argument("--duration", type=float, help="Recording duration in seconds for microphone mode.")
    _ = parser.add_argument("--output", help="Recording output WAV path; must be under repo .artifacts/.")
    _ = parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to a .mlmodel, .mlpackage, or .mlmodelc Silero VAD model.",
    )
    _ = parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Inclusive speech threshold.")
    _ = parser.add_argument("--json-output", help="Optional path to write JSON summary.")
    _ = parser.add_argument("--dry-run", action="store_true", help="Print the plan only; do not record or analyze.")
    return parser


def parse_args(argv: Sequence[str], *, repo_root: Path | None = None) -> ParsedArgs:
    root = (repo_root or repo_root_from_script()).resolve()
    parser = build_parser()
    namespace = cast(ParsedArgsNamespace, cast(object, parser.parse_args(argv)))

    has_input = namespace.input_wav is not None
    has_duration = namespace.duration is not None
    has_output = namespace.output is not None
    has_recording_mode = has_duration or has_output

    if has_input and has_recording_mode:
        parser.error("choose either --input-wav or --duration/--output recording mode, not both")
    if not has_input and not has_recording_mode:
        parser.error("choose one mode: --input-wav, or both --duration and --output")
    if has_duration != has_output:
        parser.error("recording mode requires both --duration and --output")
    if namespace.duration is not None and namespace.duration <= 0:
        parser.error("--duration must be greater than 0")
    if namespace.threshold < 0.0 or namespace.threshold > 1.0:
        parser.error("--threshold must be between 0.0 and 1.0")

    input_wav = Path(namespace.input_wav).expanduser().resolve() if namespace.input_wav else None
    output = Path(namespace.output).expanduser().resolve() if namespace.output else None
    if output is not None:
        artifacts_root = (root / ".artifacts").resolve()
        if not path_is_relative_to(output, artifacts_root):
            parser.error("--output must be under the repository .artifacts/ directory")
    model_path = Path(namespace.model_path).expanduser()
    if not model_path.is_absolute():
        model_path = root / model_path
    model_path = model_path.resolve()
    json_output = Path(namespace.json_output).expanduser().resolve() if namespace.json_output else None

    return ParsedArgs(
        input_wav=input_wav,
        duration=namespace.duration,
        output=output,
        model_path=model_path,
        threshold=namespace.threshold,
        json_output=json_output,
        dry_run=namespace.dry_run,
        repo_root=root,
    )


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        _ = path.relative_to(parent)
        return True
    except ValueError:
        return False


def recorder_command_args(name: str, executable: str, duration: float, output: Path) -> list[str]:
    if name == "sox" or name == "rec":
        return [
            executable,
            "-q",
            "-r",
            str(SAMPLE_RATE),
            "-c",
            "1",
            "-b",
            "16",
            str(output),
            "trim",
            "0",
            format_duration(duration),
        ]
    if name == "ffmpeg":
        return [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-i",
            ":0",
            "-t",
            format_duration(duration),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-y",
            str(output),
        ]
    if name == "afrecord":
        return [executable, "-f", "WAVE", "-d", format_duration(duration), "-r", str(SAMPLE_RATE), str(output)]
    raise MicrophoneValidationError(f"unsupported recorder command: {name}")


def format_duration(duration: float) -> str:
    if duration.is_integer():
        return str(int(duration))
    return f"{duration:.3f}".rstrip("0").rstrip(".")


def find_supported_recorder() -> Recorder | None:
    for name in SUPPORTED_RECORDERS:
        executable = shutil.which(name)
        if executable:
            return RecorderCommand(name, executable)
    return None


def missing_recorder_message() -> str:
    supported = ", ".join(SUPPORTED_RECORDERS)
    return (
        "No supported local recorder command was found. Use --input-wav with an existing "
        "16 kHz PCM WAV, or install/grant access to a supported local recorder "
        f"({supported}) and retry recording mode."
    )


def recording_failed_message(recorder_name: str, exc: Exception) -> str:
    return (
        f"Recorder '{recorder_name}' failed: {exc}. On macOS, open System Settings -> "
        "Privacy & Security -> Microphone, grant microphone access to Terminal/iTerm, "
        "rerun this command, or use --input-wav with an existing recording."
    )


def print_recording_consent(*, duration: float, output: Path, printer: Callable[[str], None]) -> None:
    printer("Microphone recording requested explicitly.")
    printer(f"Duration: {duration:.3f} seconds")
    printer(f"Output: {output}")
    printer("Speak during the recording window. No recording starts unless --duration and --output are both present.")


def print_plan(args: ParsedArgs, *, printer: Callable[[str], None]) -> None:
    source = str(args.input_wav) if args.input_wav is not None else str(args.output)
    mode = "existing_wav" if args.input_wav is not None else "recording"
    printer("Microphone VAD validation dry-run plan")
    printer(f"mode={mode}")
    if args.duration is not None:
        printer(f"duration_seconds={args.duration:.3f}")
    printer(f"source={source}")
    printer(f"model_path={args.model_path}")
    printer(f"threshold={args.threshold:.3f}")
    if args.json_output is not None:
        printer(f"json_output={args.json_output}")
    printer("dry_run=true: no recording, CoreML loading, WAV loading, or analysis will run")


def import_evaluate_ten_vad() -> EvaluateTenVadModuleLike:
    return cast(EvaluateTenVadModuleLike, cast(object, importlib.import_module("evaluate_ten_vad")))


def load_wav_samples(path: Path) -> list[float]:
    try:
        evaluate = import_evaluate_ten_vad()
    except ImportError as exc:
        raise MicrophoneValidationError("Could not import Scripts/evaluate_ten_vad.py helper module.") from exc

    try:
        return evaluate.load_pcm_wave_mono_16khz_stdlib(path)
    except Exception as exc:
        raise MicrophoneValidationError(str(exc)) from exc


def load_coreml_runtime(model_path: Path) -> tuple[ValidateCoreMLModuleLike, object, object]:
    try:
        np_module = importlib.import_module("numpy")
        ct_module = importlib.import_module("coremltools")
        validate_module = cast(
            ValidateCoreMLModuleLike,
            cast(object, importlib.import_module("validate_coreml")),
        )
    except ImportError as exc:
        raise MicrophoneValidationError(
            "Missing Python dependencies for CoreML prediction. "
            + "Install the optional validator dependencies referenced by Scripts/requirements.txt, then retry."
        ) from exc

    coreml_model = validate_module.load_coreml_model(ct_module, model_path)
    validate_module.validate_coreml_signature(coreml_model)
    return validate_module, np_module, coreml_model


def predict_probabilities(
    *,
    frames: Sequence[Sequence[float]],
    validate_module: ValidateCoreMLModuleLike,
    np_module: object,
    coreml_model: object,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[float], list[float]]:
    evaluate = import_evaluate_ten_vad()
    state = validate_module.zero_state(np_module)
    context = validate_module.zero_context(np_module)
    probabilities: list[float] = []
    latencies: list[float] = []

    for frame in frames:
        def call_predict() -> tuple[float, object, object]:
            return validate_module.predict_coreml(coreml_model, np_module, frame, state, context)

        probability, state, context = evaluate.time_call(call_predict, clock=clock, latencies=latencies)
        probabilities.append(float(probability))

    return probabilities, latencies


def summarize_segments(predictions: Sequence[bool], *, hop_size: int = HOP_SIZE, sample_rate: int = SAMPLE_RATE) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    start_index: int | None = None
    for index, is_speech in enumerate(predictions):
        if is_speech and start_index is None:
            start_index = index
        if (not is_speech or index == len(predictions) - 1) and start_index is not None:
            end_index = index + 1 if is_speech and index == len(predictions) - 1 else index
            start = start_index * hop_size / sample_rate
            end = end_index * hop_size / sample_rate
            segments.append(SpeechSegment(start=start, end=end, duration=end - start, frames=end_index - start_index))
            start_index = None
    return segments


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def probability_stats(probabilities: Sequence[float]) -> dict[str, float]:
    if not probabilities:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(probabilities),
        "mean": sum(probabilities) / len(probabilities),
        "max": max(probabilities),
    }


def summarize_vad(
    *,
    source: Path,
    probabilities: Sequence[float],
    latencies: Sequence[float],
    threshold: float,
) -> VadSummary:
    predictions = [probability >= threshold for probability in probabilities]
    frames = len(probabilities)
    audio_seconds = frames * HOP_SIZE / SAMPLE_RATE
    speech_frames = sum(1 for prediction in predictions if prediction)
    speech_seconds = speech_frames * HOP_SIZE / SAMPLE_RATE
    inference_seconds = sum(latencies)
    latency_summary = summarize_latency(latencies)
    timing: dict[str, float | int] = {
        "inference_seconds": inference_seconds,
        "rtf": safe_ratio(inference_seconds, audio_seconds),
        "rtfx": safe_ratio(audio_seconds, inference_seconds),
        **latency_summary,
    }
    return VadSummary(
        source=str(source),
        threshold=threshold,
        frames=frames,
        audio_seconds=audio_seconds,
        speech_frames=speech_frames,
        speech_seconds=speech_seconds,
        speech_ratio=safe_ratio(speech_frames, frames),
        segments=summarize_segments(predictions),
        probability=probability_stats(probabilities),
        timing=timing,
    )


def summarize_latency(latencies_seconds: Sequence[float]) -> dict[str, float | int]:
    try:
        evaluate = import_evaluate_ten_vad()
        return evaluate.summarize_latency(latencies_seconds)
    except ImportError:
        values_ms = sorted(value * 1000.0 for value in latencies_seconds)
        count = len(values_ms)
        return {
            "count": count,
            "mean_ms": safe_ratio(sum(values_ms), count),
            "p50_ms": percentile(values_ms, 0.50),
            "p95_ms": percentile(values_ms, 0.95),
            "max_ms": values_ms[-1] if values_ms else 0.0,
        }


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def run_analysis(
    *,
    wav_path: Path,
    model_path: Path,
    threshold: float,
    wav_loader: Callable[[Path], list[float]] = load_wav_samples,
    runtime_loader: Callable[[Path], tuple[ValidateCoreMLModuleLike, object, object]] = load_coreml_runtime,
    clock: Callable[[], float] = time.perf_counter,
) -> VadSummary:
    samples = wav_loader(wav_path)
    frames = [list(frame) for frame in chunk_audio_frames(samples, chunk_size=HOP_SIZE)]
    validate_module, np_module, coreml_model = runtime_loader(model_path)
    probabilities, latencies = predict_probabilities(
        frames=frames,
        validate_module=validate_module,
        np_module=np_module,
        coreml_model=coreml_model,
        clock=clock,
    )
    return summarize_vad(source=wav_path, probabilities=probabilities, latencies=latencies, threshold=threshold)


def chunk_audio_frames(samples: Sequence[float], *, chunk_size: int) -> list[list[float]]:
    try:
        evaluate = import_evaluate_ten_vad()
        return evaluate.chunk_audio_frames(samples, chunk_size=chunk_size)
    except ImportError:
        complete = len(samples) // chunk_size
        return [list(samples[index * chunk_size : (index + 1) * chunk_size]) for index in range(complete)]


def print_summary(summary: VadSummary, *, printer: Callable[[str], None]) -> None:
    printer(
        f"source={summary.source} frames={summary.frames} audio_seconds={summary.audio_seconds:.3f} "
        + f"threshold={summary.threshold:.3f} speech_frames={summary.speech_frames} "
        + f"speech_seconds={summary.speech_seconds:.3f} speech_ratio={summary.speech_ratio:.6f}"
    )
    printer(
        f"probability min={summary.probability['min']:.6f} "
        + f"mean={summary.probability['mean']:.6f} max={summary.probability['max']:.6f}"
    )
    printer(
        f"timing inference_seconds={float(summary.timing['inference_seconds']):.6f} "
        + f"rtf={float(summary.timing['rtf']):.6f} rtfx={float(summary.timing['rtfx']):.3f} "
        + f"latency_mean_ms={float(summary.timing['mean_ms']):.3f} "
        + f"p50_ms={float(summary.timing['p50_ms']):.3f} "
        + f"p95_ms={float(summary.timing['p95_ms']):.3f} "
        + f"max_ms={float(summary.timing['max_ms']):.3f}"
    )
    if summary.segments:
        for index, segment in enumerate(summary.segments, start=1):
            printer(
                f"segment={index} start={segment.start:.3f} end={segment.end:.3f} "
                + f"duration={segment.duration:.3f} frames={segment.frames}"
            )
    else:
        printer("segments=none")


def summary_to_jsonable(summary: VadSummary) -> dict[str, object]:
    return asdict(summary)


def main(
    argv: Sequence[str],
    *,
    recorder_factory: Callable[[], Recorder | None] = find_supported_recorder,
    wav_loader: Callable[[Path], list[float]] = load_wav_samples,
    runtime_loader: Callable[[Path], tuple[ValidateCoreMLModuleLike, object, object]] = load_coreml_runtime,
    printer: Callable[[str], None] = print,
    repo_root: Path | None = None,
) -> int:
    try:
        args = parse_args(argv, repo_root=repo_root)
        if args.dry_run:
            print_plan(args, printer=printer)
            return 0

        wav_path = args.input_wav
        if wav_path is None:
            assert args.duration is not None
            assert args.output is not None
            print_recording_consent(duration=args.duration, output=args.output, printer=printer)
            recorder = recorder_factory()
            if recorder is None:
                raise MicrophoneValidationError(missing_recorder_message())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            recorder.record(duration=args.duration, output=args.output)
            wav_path = args.output

        summary = run_analysis(
            wav_path=wav_path,
            model_path=args.model_path,
            threshold=args.threshold,
            wav_loader=wav_loader,
            runtime_loader=runtime_loader,
        )
        print_summary(summary, printer=printer)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            _ = args.json_output.write_text(
                json.dumps(summary_to_jsonable(summary), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return 0
    except MicrophoneValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
