#!/usr/bin/env python3
"""Evaluate a local CoreML Silero VAD model against local TEN-VAD fixtures.

This optional evaluator does not vendor TEN audio/labels and never downloads
fixtures by default. It keeps imports lightweight so ``--help`` and unit tests
work without CoreMLTools, numpy, torchaudio, or network access.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
import wave
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, TypeVar, cast


SAMPLE_RATE = 16_000
HOP_SIZE = 512
TEN_TESTSET_COMMIT = "22a3bcd4509d0faaa8eef4881e8af5f39c178950"
DEFAULT_THRESHOLD = 0.5
DEFAULT_THRESHOLD_STEP = 0.01
ALLOWED_MODEL_SUFFIXES = {".mlmodel", ".mlpackage", ".mlmodelc"}
DEFAULT_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / ".artifacts"
    / "ten-vad-testset"
    / TEN_TESTSET_COMMIT
    / "testset"
)


class EvaluationError(Exception):
    pass


@dataclass(frozen=True)
class LabelSegment:
    start: float
    end: float
    label: int


@dataclass(frozen=True)
class TenAnnotations:
    filename: str
    segments: list[LabelSegment]


@dataclass(frozen=True)
class FixturePair:
    stem: str
    wav_path: Path
    scv_path: Path


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0


@dataclass(frozen=True)
class FileEvaluation:
    filename: str
    frames: int
    label_frames: int
    prediction_frames: int
    aligned_frames: int
    dropped_label_frames: int
    dropped_prediction_frames: int
    audio_seconds: float
    threshold_metrics: dict[str, float | int]


@dataclass(frozen=True)
class EvaluationResult:
    files: list[FileEvaluation]
    overall: dict[str, float | int]
    sweep: list[dict[str, object]]
    timing: dict[str, float | int]


class WaveformLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    def dim(self) -> int: ...

    def mean(self, *, dim: int, keepdim: bool) -> "WaveformLike": ...

    def reshape(self, *shape: int) -> "WaveformLike": ...

    def tolist(self) -> list[float]: ...


class TorchaudioFunctionalLike(Protocol):
    def resample(self, waveform: WaveformLike, orig_freq: int, new_freq: int) -> WaveformLike: ...


class TorchaudioModuleLike(Protocol):
    functional: TorchaudioFunctionalLike

    def load(self, uri: str) -> tuple[WaveformLike, int]: ...


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


class ParsedNamespace(argparse.Namespace):
    model_path: str | None = None
    ten_testset_dir: str | None = None
    threshold_step: float = DEFAULT_THRESHOLD_STEP
    thresholds: str | None = None
    summary_threshold: float = DEFAULT_THRESHOLD
    json_output: str | None = None
    limit_files: int | None = None
    print_fixture_download_plan: bool = False
    download_fixtures: bool = False


@dataclass(frozen=True)
class ParsedArgs:
    print_fixture_download_plan: bool
    model_path: Path | None
    ten_testset_dir: Path
    threshold_step: float
    thresholds: str | None
    summary_threshold: float
    json_output: Path | None
    limit_files: int | None


_T = TypeVar("_T")


def numeric_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise EvaluationError(f"expected numeric metric value, got {type(value).__name__}")


def parse_scv_line(line: str) -> TenAnnotations:
    fields = [field.strip() for field in line.strip().split(",")]
    if not fields or fields == [""]:
        raise EvaluationError("TEN .scv line is empty")

    filename = fields[0]
    if not filename:
        raise EvaluationError("TEN .scv filename is empty")

    values = fields[1:]
    if len(values) % 3 != 0:
        raise EvaluationError("TEN .scv fields after filename must be divisible triples")

    segments: list[LabelSegment] = []
    previous_end = 0.0
    for index in range(0, len(values), 3):
        raw_start, raw_end, raw_label = values[index : index + 3]
        try:
            start = float(raw_start)
            end = float(raw_end)
        except ValueError as exc:
            raise EvaluationError("TEN .scv segment times must be numeric") from exc

        if raw_label not in {"0", "1"}:
            raise EvaluationError("TEN .scv label must be 0 or 1")
        label = int(raw_label)

        if start < 0 or end < 0:
            raise EvaluationError("TEN .scv segment times must be non-negative")
        if end < start:
            raise EvaluationError("TEN .scv segment times must be monotonic")
        if segments and start < previous_end:
            raise EvaluationError("TEN .scv segment times must be monotonic")

        segments.append(LabelSegment(start=start, end=end, label=label))
        previous_end = end

    return TenAnnotations(filename=filename, segments=segments)


def parse_scv_file(path: Path) -> TenAnnotations:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise EvaluationError(f"TEN .scv file must contain exactly one non-empty line: {path}")
    annotations = parse_scv_line(lines[0])
    if annotations.filename != path.stem:
        raise EvaluationError(
            f"TEN .scv filename '{annotations.filename}' does not match file stem '{path.stem}'"
        )
    return annotations


def expand_labels(
    segments: Sequence[LabelSegment],
    *,
    hop_size: int = HOP_SIZE,
    sample_rate: int = SAMPLE_RATE,
) -> list[int]:
    frame_duration = hop_size / sample_rate
    labels: list[int] = []
    for segment in segments:
        duration = segment.end - segment.start
        frame_count = round(duration / frame_duration)
        labels.extend([segment.label] * frame_count)
    return labels


def chunk_audio_frames(samples: Sequence[float], *, chunk_size: int = HOP_SIZE) -> list[list[float]]:
    complete = len(samples) // chunk_size
    return [list(samples[index * chunk_size : (index + 1) * chunk_size]) for index in range(complete)]


def threshold_probabilities(probabilities: Sequence[float], *, threshold: float) -> list[int]:
    return [1 if probability >= threshold else 0 for probability in probabilities]


def confusion_counts(labels: Sequence[int], predictions: Sequence[int]) -> ConfusionCounts:
    tp = tn = fp = fn = 0
    for label, prediction in zip(labels, predictions):
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1 and prediction == 0:
            fn += 1
        else:
            raise EvaluationError("labels and predictions must contain only 0 or 1")
    return ConfusionCounts(tp=tp, tn=tn, fp=fp, fn=fn)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def metrics_from_counts(counts: ConfusionCounts) -> dict[str, float]:
    precision = safe_ratio(counts.tp, counts.tp + counts.fp)
    recall = safe_ratio(counts.tp, counts.tp + counts.fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_ratio(2 * precision * recall, precision + recall),
        "accuracy": safe_ratio(counts.tp + counts.tn, counts.tp + counts.tn + counts.fp + counts.fn),
        "fpr": safe_ratio(counts.fp, counts.fp + counts.tn),
        "fnr": safe_ratio(counts.fn, counts.fn + counts.tp),
    }


def build_thresholds(*, step: float, explicit_thresholds: str | None) -> list[float]:
    if explicit_thresholds:
        thresholds: set[float] = set()
        for raw_value in explicit_thresholds.split(","):
            try:
                threshold = float(raw_value.strip())
            except ValueError as exc:
                raise EvaluationError(f"invalid threshold: {raw_value}") from exc
            if threshold < 0.0 or threshold > 1.0:
                raise EvaluationError("thresholds must be between 0.0 and 1.0")
            thresholds.add(round(threshold, 10))
        if not thresholds:
            raise EvaluationError("--thresholds must include at least one value")
        return sorted(thresholds)

    if step <= 0.0 or step > 1.0:
        raise EvaluationError("--threshold-step must be greater than 0.0 and at most 1.0")

    thresholds = {0.0, 1.0}
    value = 0.0
    while value < 1.0:
        thresholds.add(round(value, 10))
        value += step
    return sorted(thresholds)


def threshold_sweep(
    *,
    labels: Sequence[int],
    probabilities: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        predictions = threshold_probabilities(probabilities, threshold=threshold)
        counts = confusion_counts(labels, predictions)
        rows.append(
            {
                "threshold": threshold,
                "counts": asdict(counts),
                **metrics_from_counts(counts),
            }
        )
    return rows


def alignment_accounting(
    label_frames: int,
    prediction_frames: int,
) -> dict[str, int]:
    aligned_frames = min(label_frames, prediction_frames)
    return {
        "label_frames": label_frames,
        "prediction_frames": prediction_frames,
        "aligned_frames": aligned_frames,
        "dropped_label_frames": label_frames - aligned_frames,
        "dropped_prediction_frames": prediction_frames - aligned_frames,
    }


def should_warn_alignment_mismatch(label_frames: int, prediction_frames: int) -> bool:
    mismatch = abs(label_frames - prediction_frames)
    max_frames = max(label_frames, prediction_frames)
    return mismatch > 1 and mismatch > max_frames * 0.005


def format_alignment_warning(filename: str, accounting: dict[str, int]) -> str:
    mismatch = abs(accounting["label_frames"] - accounting["prediction_frames"])
    return (
        f"warning: file={filename} alignment mismatch mismatch_frames={mismatch} "
        f"label_frames={accounting['label_frames']} "
        f"prediction_frames={accounting['prediction_frames']} "
        f"aligned_frames={accounting['aligned_frames']} "
        f"dropped_label_frames={accounting['dropped_label_frames']} "
        f"dropped_prediction_frames={accounting['dropped_prediction_frames']}"
    )


def time_call(
    callback: Callable[[], _T],
    *,
    clock: Callable[[], float] = time.perf_counter,
    latencies: list[float],
) -> _T:
    start = clock()
    result = callback()
    end = clock()
    latencies.append(end - start)
    return result


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


def summarize_latency(latencies_seconds: Sequence[float]) -> dict[str, float | int]:
    values_ms = sorted(value * 1000.0 for value in latencies_seconds)
    count = len(values_ms)
    return {
        "count": count,
        "mean_ms": safe_ratio(sum(values_ms), count),
        "p50_ms": percentile(values_ms, 0.50),
        "p95_ms": percentile(values_ms, 0.95),
        "max_ms": values_ms[-1] if values_ms else 0.0,
    }


def fixture_download_plan(artifacts_root: Path) -> dict[str, object]:
    destination = artifacts_root / "ten-vad-testset" / TEN_TESTSET_COMMIT / "testset"
    return {
        "commit": TEN_TESTSET_COMMIT,
        "destination": destination,
        "expected_pairs": 30,
        "expected_names": [f"testset-audio-{index:02d}" for index in range(1, 31)],
        "note": (
            "This command performs no network access. Download TEN-VAD testset "
            "audio/labels explicitly from the pinned commit and place only local "
            "copies under the ignored .artifacts/ directory."
        ),
    }


def fixture_missing_message(testset_dir: Path) -> str:
    return (
        "TEN fixtures are missing. This optional evaluator does not download audio "
        "or labels by default. Provide --ten-testset-dir pointing at local TEN "
        ".wav/.scv pairs, or run --print-fixture-download-plan for the pinned "
        f"commit {TEN_TESTSET_COMMIT}. Expected default path: {testset_dir}"
    )


def find_fixture_pairs(testset_dir: Path, *, limit_files: int | None) -> list[FixturePair]:
    if not testset_dir.exists():
        raise EvaluationError(fixture_missing_message(testset_dir))
    if not testset_dir.is_dir():
        raise EvaluationError(f"TEN testset path is not a directory: {testset_dir}")

    pairs: list[FixturePair] = []
    for scv_path in sorted(testset_dir.glob("*.scv")):
        wav_path = scv_path.with_suffix(".wav")
        if wav_path.exists():
            pairs.append(FixturePair(stem=scv_path.stem, wav_path=wav_path, scv_path=scv_path))

    if not pairs:
        raise EvaluationError(fixture_missing_message(testset_dir))
    if limit_files is not None:
        pairs = pairs[:limit_files]
    return pairs


def load_torchaudio_module() -> TorchaudioModuleLike:
    try:
        module = importlib.import_module("torchaudio")
        return cast(TorchaudioModuleLike, cast(object, module))
    except ImportError as exc:
        raise EvaluationError(
            "Missing Python dependency: torchaudio. Install the optional evaluator dependencies referenced by Scripts/requirements.txt, then retry."
        ) from exc


def should_fallback_to_stdlib_wav(path: Path, exc: Exception) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    message = str(exc).lower()
    return (
        "backend" in message
        and (
            "appropriate backend" in message
            or "handle uri" in message
            or "no audio backend" in message
        )
    )


def pcm_sample_to_float(sample: bytes, *, sample_width: int) -> float:
    if sample_width == 1:
        return (sample[0] - 128) / 128.0
    if sample_width == 3:
        sign_extension = b"\xff" if sample[2] & 0x80 else b"\x00"
        value = int.from_bytes(sample + sign_extension, byteorder="little", signed=True)
        return value / float(1 << 23)
    value = int.from_bytes(sample, byteorder="little", signed=True)
    return value / float(1 << (sample_width * 8 - 1))


def load_pcm_wave_mono_16khz_stdlib(path: Path) -> list[float]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            compression_type = wav_file.getcomptype()
            frame_count = wav_file.getnframes()
            raw_frames = wav_file.readframes(frame_count)
    except wave.Error as exc:
        raise EvaluationError(f"Python stdlib WAV fallback could not read {path}: {exc}") from exc

    if compression_type != "NONE":
        raise EvaluationError(
            f"Python stdlib WAV fallback only supports uncompressed PCM WAV files; {path} uses compression type {compression_type!r}."
        )
    if sample_rate != SAMPLE_RATE:
        raise EvaluationError(
            f"Python stdlib WAV fallback only supports 16000 Hz PCM WAV files when torchaudio decoding is unavailable; got {sample_rate} Hz for {path}."
        )
    if sample_width not in {1, 2, 3, 4}:
        raise EvaluationError(
            f"Python stdlib WAV fallback only supports 8-, 16-, 24-, or 32-bit PCM integer WAV files; got {sample_width * 8}-bit data in {path}."
        )
    if channels < 1:
        raise EvaluationError(f"Python stdlib WAV fallback expected at least one channel in {path}.")

    bytes_per_frame = sample_width * channels
    if bytes_per_frame == 0 or len(raw_frames) % bytes_per_frame != 0:
        raise EvaluationError(f"Python stdlib WAV fallback found truncated PCM frame data in {path}.")

    samples: list[float] = []
    raw_view = memoryview(raw_frames)
    for frame_offset in range(0, len(raw_view), bytes_per_frame):
        channel_sum = 0.0
        for channel_index in range(channels):
            sample_offset = frame_offset + channel_index * sample_width
            sample = bytes(raw_view[sample_offset : sample_offset + sample_width])
            channel_sum += pcm_sample_to_float(sample, sample_width=sample_width)
        samples.append(channel_sum / channels)
    return samples


def load_wav_mono_16khz(path: Path, *, torchaudio_module: TorchaudioModuleLike) -> list[float]:
    try:
        waveform, sample_rate = torchaudio_module.load(str(path))
    except Exception as exc:
        if not should_fallback_to_stdlib_wav(path, exc):
            raise
        return load_pcm_wave_mono_16khz_stdlib(path)
    if waveform.dim() != 2:
        raise EvaluationError(f"expected waveform [channels, samples], got shape {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio_module.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return [float(value) for value in waveform.reshape(-1).tolist()]


def load_coreml_runtime(model_path: Path) -> tuple[ValidateCoreMLModuleLike, object, object]:
    try:
        np_module = cast(object, importlib.import_module("numpy"))
        ct_module = cast(object, importlib.import_module("coremltools"))
        validate_module = importlib.import_module("validate_coreml")
        validate = cast(ValidateCoreMLModuleLike, cast(object, validate_module))
    except ImportError as exc:
        raise EvaluationError(
            "Missing Python dependencies for CoreML prediction. Install the optional evaluator dependencies referenced by Scripts/requirements.txt, then retry."
        ) from exc

    coreml_model = validate.load_coreml_model(ct_module, model_path)
    validate.validate_coreml_signature(coreml_model)
    return validate, np_module, coreml_model


def predict_file_probabilities(
    *,
    frames: Sequence[Sequence[float]],
    validate_module: ValidateCoreMLModuleLike,
    np_module: object,
    coreml_model: object,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[float], list[float]]:
    state = validate_module.zero_state(np_module)
    context = validate_module.zero_context(np_module)
    probabilities: list[float] = []
    latencies: list[float] = []

    for frame in frames:
        def call_predict() -> tuple[float, object, object]:
            return validate_module.predict_coreml(
                coreml_model,
                np_module,
                frame,
                state,
                context,
            )
        prediction = time_call(call_predict, clock=clock, latencies=latencies)
        probability, state, context = prediction
        probabilities.append(float(probability))

    return probabilities, latencies


def select_overall_threshold(sweep: Sequence[dict[str, object]], threshold: float) -> dict[str, float | int]:
    for row in sweep:
        row_threshold = numeric_value(row["threshold"])
        if math.isclose(row_threshold, threshold, abs_tol=1e-10):
            counts = cast(dict[str, int], row["counts"])
            flattened: dict[str, float | int] = {"threshold": row_threshold}
            flattened.update({key: int(value) for key, value in counts.items()})
            for key in ("precision", "recall", "f1", "accuracy", "fpr", "fnr"):
                flattened[key] = numeric_value(row[key])
            return flattened
    raise EvaluationError(f"threshold {threshold} is not included in the threshold sweep")


def run_evaluation(
    *,
    model_path: Path,
    testset_dir: Path,
    thresholds: Sequence[float],
    summary_threshold: float = DEFAULT_THRESHOLD,
    limit_files: int | None = None,
    printer: Callable[[str], None] = print,
) -> EvaluationResult:
    pairs = find_fixture_pairs(testset_dir, limit_files=limit_files)
    torchaudio_module = load_torchaudio_module()
    validate_module, np_module, coreml_model = load_coreml_runtime(model_path)

    all_labels: list[int] = []
    all_probabilities: list[float] = []
    files: list[FileEvaluation] = []
    all_latencies: list[float] = []
    audio_seconds = 0.0

    for pair in pairs:
        annotations = parse_scv_file(pair.scv_path)
        samples = load_wav_mono_16khz(pair.wav_path, torchaudio_module=torchaudio_module)
        frames = chunk_audio_frames(samples, chunk_size=HOP_SIZE)
        labels = expand_labels(annotations.segments, hop_size=HOP_SIZE, sample_rate=SAMPLE_RATE)
        probabilities, latencies = predict_file_probabilities(
            frames=frames,
            validate_module=validate_module,
            np_module=np_module,
            coreml_model=coreml_model,
        )
        accounting = alignment_accounting(len(labels), len(probabilities))
        if should_warn_alignment_mismatch(accounting["label_frames"], accounting["prediction_frames"]):
            printer(format_alignment_warning(pair.stem, accounting))

        aligned_labels = labels[: accounting["aligned_frames"]]
        aligned_probabilities = probabilities[: accounting["aligned_frames"]]
        all_labels.extend(aligned_labels)
        all_probabilities.extend(aligned_probabilities)
        all_latencies.extend(latencies[: accounting["aligned_frames"]])

        file_seconds = accounting["aligned_frames"] * HOP_SIZE / SAMPLE_RATE
        audio_seconds += file_seconds
        file_sweep = threshold_sweep(
            labels=aligned_labels,
            probabilities=aligned_probabilities,
            thresholds=thresholds,
        )
        file_metrics = select_overall_threshold(file_sweep, summary_threshold)
        files.append(
            FileEvaluation(
                filename=pair.stem,
                frames=accounting["aligned_frames"],
                label_frames=accounting["label_frames"],
                prediction_frames=accounting["prediction_frames"],
                aligned_frames=accounting["aligned_frames"],
                dropped_label_frames=accounting["dropped_label_frames"],
                dropped_prediction_frames=accounting["dropped_prediction_frames"],
                audio_seconds=file_seconds,
                threshold_metrics=file_metrics,
            )
        )
        file_summary = (
            f"file={pair.stem} frames={accounting['aligned_frames']} "
            f"label_frames={accounting['label_frames']} "
            f"prediction_frames={accounting['prediction_frames']} "
            f"aligned_frames={accounting['aligned_frames']} "
            f"dropped_label_frames={accounting['dropped_label_frames']} "
            f"dropped_prediction_frames={accounting['dropped_prediction_frames']} "
            f"audio_seconds={file_seconds:.3f} "
            f"threshold={summary_threshold:.3f} f1={file_metrics['f1']:.6f} "
            f"precision={file_metrics['precision']:.6f} recall={file_metrics['recall']:.6f}"
        )
        printer(file_summary)

    sweep = threshold_sweep(labels=all_labels, probabilities=all_probabilities, thresholds=thresholds)
    overall = select_overall_threshold(sweep, summary_threshold)
    inference_seconds = sum(all_latencies)
    timing = {
        "inference_seconds": inference_seconds,
        "audio_seconds": audio_seconds,
        "rtf": safe_ratio(inference_seconds, audio_seconds),
        "rtfx": safe_ratio(audio_seconds, inference_seconds),
        **summarize_latency(all_latencies),
    }
    overall_summary = (
        f"overall frames={len(all_labels)} audio_seconds={audio_seconds:.3f} "
        f"threshold={summary_threshold:.3f} f1={overall['f1']:.6f} "
        f"precision={overall['precision']:.6f} recall={overall['recall']:.6f} "
        f"rtf={timing['rtf']:.6f} rtfx={timing['rtfx']:.3f}"
    )
    printer(overall_summary)
    return EvaluationResult(files=files, overall=overall, sweep=sweep, timing=timing)


def result_to_jsonable(result: EvaluationResult) -> dict[str, object]:
    return {
        "files": [asdict(file_result) for file_result in result.files],
        "overall": result.overall,
        "sweep": result.sweep,
        "timing": result.timing,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate local CoreML Silero VAD output against local TEN-VAD .wav/.scv fixtures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--model-path", help="Path to a .mlmodel, .mlpackage, or .mlmodelc model.")
    _ = parser.add_argument(
        "--ten-testset-dir",
        help="Explicit local TEN testset directory containing .wav/.scv pairs.",
    )
    _ = parser.add_argument(
        "--threshold-step",
        type=float,
        default=DEFAULT_THRESHOLD_STEP,
        help="Threshold sweep step; endpoints 0.0 and 1.0 are always included.",
    )
    _ = parser.add_argument(
        "--thresholds",
        help="Comma-separated explicit threshold list, overriding --threshold-step.",
    )
    _ = parser.add_argument(
        "--summary-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold used for printed per-file and overall summaries.",
    )
    _ = parser.add_argument("--json-output", help="Optional path to write JSON results.")
    _ = parser.add_argument(
        "--limit-files",
        type=int,
        help="Limit number of fixture pairs for quick local runs.",
    )
    _ = parser.add_argument(
        "--print-fixture-download-plan",
        action="store_true",
        help="Print the explicit pinned local fixture acquisition plan and exit; performs no network access.",
    )
    _ = parser.add_argument(
        "--download-fixtures",
        action="store_true",
        help="Reserved explicit download flag. Not implemented; use --print-fixture-download-plan.",
    )
    return parser


def parse_args(argv: Sequence[str]) -> ParsedArgs:
    parser = build_parser()
    namespace = parser.parse_args(argv, namespace=ParsedNamespace())

    if namespace.print_fixture_download_plan:
        return ParsedArgs(
            print_fixture_download_plan=True,
            model_path=None,
            ten_testset_dir=DEFAULT_FIXTURE_DIR,
            threshold_step=namespace.threshold_step,
            thresholds=namespace.thresholds,
            summary_threshold=namespace.summary_threshold,
            json_output=None,
            limit_files=namespace.limit_files,
        )

    if namespace.download_fixtures:
        parser.error("--download-fixtures is not implemented; use --print-fixture-download-plan")

    if not namespace.model_path:
        parser.error("--model-path is required unless --print-fixture-download-plan is used")

    model_path = Path(namespace.model_path).expanduser().resolve()
    if not model_path.exists():
        parser.error(f"model path does not exist: {model_path}")
    if model_path.suffix not in ALLOWED_MODEL_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_MODEL_SUFFIXES))
        parser.error(f"model path must end with one of {{{allowed}}}, got: {model_path.name}")

    if namespace.limit_files is not None and namespace.limit_files < 1:
        parser.error("--limit-files must be at least 1")
    if namespace.summary_threshold < 0.0 or namespace.summary_threshold > 1.0:
        parser.error("--summary-threshold must be between 0.0 and 1.0")

    ten_testset_dir = (
        Path(namespace.ten_testset_dir).expanduser().resolve()
        if namespace.ten_testset_dir
        else DEFAULT_FIXTURE_DIR
    )
    json_output = Path(namespace.json_output).expanduser().resolve() if namespace.json_output else None
    return ParsedArgs(
        print_fixture_download_plan=False,
        model_path=model_path,
        ten_testset_dir=ten_testset_dir,
        threshold_step=namespace.threshold_step,
        thresholds=namespace.thresholds,
        summary_threshold=namespace.summary_threshold,
        json_output=json_output,
        limit_files=namespace.limit_files,
    )


def main(argv: Sequence[str]) -> int:
    try:
        args = parse_args(argv)
        if args.print_fixture_download_plan:
            print(json.dumps(fixture_download_plan(Path.cwd() / ".artifacts"), indent=2, default=str))
            return 0

        thresholds = build_thresholds(step=args.threshold_step, explicit_thresholds=args.thresholds)
        if not any(math.isclose(value, args.summary_threshold, abs_tol=1e-10) for value in thresholds):
            thresholds = sorted({*thresholds, round(args.summary_threshold, 10)})

        model_path = args.model_path
        assert model_path is not None
        result = run_evaluation(
            model_path=model_path,
            testset_dir=args.ten_testset_dir,
            thresholds=thresholds,
            summary_threshold=args.summary_threshold,
            limit_files=args.limit_files,
        )
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            _ = args.json_output.write_text(
                json.dumps(result_to_jsonable(result), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return 0
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    _ = main(sys.argv[1:])
    raise SystemExit(_)
