from __future__ import annotations

import contextlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Protocol, cast


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class ParsedArgsLike(Protocol):
    output: Path | None
    duration: float | None


class SpeechSegmentLike(Protocol):
    start: float
    end: float
    frames: int


class VadSummaryLike(Protocol):
    frames: int
    speech_frames: int
    audio_seconds: float
    speech_seconds: float
    segments: list[SpeechSegmentLike]
    probability: dict[str, float]


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


class RecorderLike(Protocol):
    name: str

    def record(self, *, duration: float, output: Path) -> None: ...


RecorderFactory = Callable[[], RecorderLike | None]
WavLoader = Callable[[Path], list[float]]
RuntimeLoader = Callable[[Path], tuple[ValidateCoreMLModuleLike, object, object]]
Printer = Callable[[str], None]
Clock = Callable[[], float]


class MicrophoneModuleLike(Protocol):
    MicrophoneValidationError: type[Exception]

    def recording_failed_message(self, recorder_name: str, exc: Exception) -> str: ...

    def parse_args(self, argv: Sequence[str], *, repo_root: Path | None = None) -> ParsedArgsLike: ...

    def main(
        self,
        argv: Sequence[str],
        *,
        recorder_factory: RecorderFactory = ...,
        wav_loader: WavLoader = ...,
        runtime_loader: RuntimeLoader = ...,
        printer: Printer = ...,
        repo_root: Path | None = None,
    ) -> int: ...

    def run_analysis(
        self,
        *,
        wav_path: Path,
        model_path: Path,
        threshold: float,
        wav_loader: WavLoader = ...,
        runtime_loader: RuntimeLoader = ...,
        clock: Clock = ...,
    ) -> VadSummaryLike: ...


microphone = cast(
    MicrophoneModuleLike,
    cast(object, importlib.import_module("validate_microphone_vad")),
)


def _coerce_float(value: object) -> float:
    return float(cast(int | float, value))


def parse_json_object(text: str) -> dict[str, object]:
    payload = cast(object, json.loads(text))
    if not isinstance(payload, dict):
        raise AssertionError("expected JSON object")
    return cast(dict[str, object], payload)


def silent_printer(line: str) -> None:
    _ = line


def make_constant_wav_loader(samples: list[float]) -> WavLoader:
    def loader(path: Path) -> list[float]:
        _ = path
        return list(samples)

    return loader


def make_runtime_loader(validate: ValidateCoreMLModuleLike) -> RuntimeLoader:
    def loader(path: Path) -> tuple[ValidateCoreMLModuleLike, object, object]:
        _ = path
        return validate, FakeNumpy, object()

    return loader


def fail_wav_loader(path: Path) -> list[float]:
    _ = path
    raise AssertionError("wav loader should not run")


def fail_runtime_loader(path: Path) -> tuple[ValidateCoreMLModuleLike, object, object]:
    _ = path
    raise AssertionError("runtime loader should not run")


class FakeArray:
    data: list[float]
    shape: tuple[int, ...]

    def __init__(self, data: list[float], shape: tuple[int, ...]) -> None:
        self.data = list(data)
        self.shape = shape

    @property
    def size(self) -> int:
        return len(self.data)

    def reshape(self, *shape: int | tuple[int, ...]) -> "FakeArray":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            next_shape = shape[0]
        else:
            next_shape = cast(tuple[int, ...], shape)
        if next_shape == (-1,):
            next_shape = (len(self.data),)
        return FakeArray(self.data, next_shape)

    def astype(self, _dtype: object) -> "FakeArray":
        return FakeArray(self.data, self.shape)

    def __getitem__(self, index: int | slice) -> "float | FakeArray":
        if isinstance(index, slice):
            return FakeArray(self.data[index], (len(self.data[index]),))
        return self.data[index]


class FakeNumpy:
    float32: str = "float32"

    @staticmethod
    def asarray(value: object, dtype: object | None = None) -> FakeArray:
        if isinstance(value, FakeArray):
            return value.astype(dtype)
        if isinstance(value, list):
            values = cast(list[object], value)
            return FakeArray([_coerce_float(item) for item in values], (len(values),))
        if isinstance(value, tuple):
            values = cast(tuple[object, ...], value)
            return FakeArray([_coerce_float(item) for item in values], (len(values),))
        return FakeArray([_coerce_float(value)], (1,))

    @staticmethod
    def zeros(shape: tuple[int, ...] | int, _dtype: object | None = None) -> FakeArray:
        normalized = (shape,) if isinstance(shape, int) else shape
        size = 1
        for dimension in normalized:
            size *= dimension
        return FakeArray([0.0] * size, normalized)


class FakeValidateModule:
    probabilities: list[float]
    calls: list[list[float]]

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = list(probabilities)
        self.calls = []

    def load_coreml_model(self, coremltools_module: object, model_path: Path) -> object:
        _ = coremltools_module
        _ = model_path
        return object()

    def validate_coreml_signature(self, coreml_model: object) -> None:
        _ = coreml_model
        return None

    def zero_state(self, np_module: object) -> FakeArray:
        _ = np_module
        return FakeArray([0.0], (1,))

    def zero_context(self, np_module: object) -> FakeArray:
        _ = np_module
        return FakeArray([0.0] * 64, (64,))

    def predict_coreml(
        self,
        coreml_model: object,
        np_module: object,
        chunk: Sequence[float],
        state_in: object,
        context: object,
    ) -> tuple[float, FakeArray, FakeArray]:
        _ = coreml_model
        _ = np_module
        _ = state_in
        _ = context
        self.calls.append(list(chunk))
        probability = self.probabilities.pop(0)
        return probability, FakeArray([probability], (1,)), FakeArray(list(chunk[-64:]), (64,))


class FakeRecorder:
    name: str = "fake-recorder"
    should_fail: bool

    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls: list[tuple[float, Path]] = []

    def record(self, *, duration: float, output: Path) -> None:
        self.calls.append((duration, output))
        if self.should_fail:
            raise microphone.MicrophoneValidationError(microphone.recording_failed_message(self.name, RuntimeError("denied")))


class ValidateMicrophoneVadTests(unittest.TestCase):
    def parse_error(self, argv: list[str], pattern: str, *, repo_root: Path) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr):
            _ = microphone.parse_args(argv, repo_root=repo_root)
        self.assertIn(pattern, stderr.getvalue())

    def test_parser_rejects_missing_mixed_partial_duration_and_threshold_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_output = root / ".artifacts" / "microphone" / "clip.wav"
            input_wav = root / "input.wav"

            self.parse_error([], "choose one mode", repo_root=root)
            self.parse_error(["--input-wav", str(input_wav), "--duration", "1", "--output", str(artifact_output)], "choose either", repo_root=root)
            self.parse_error(["--duration", "1"], "requires both", repo_root=root)
            self.parse_error(["--output", str(artifact_output)], "requires both", repo_root=root)
            self.parse_error(["--duration", "0", "--output", str(artifact_output)], "greater than 0", repo_root=root)
            self.parse_error(["--input-wav", str(input_wav), "--threshold", "1.01"], "between 0.0 and 1.0", repo_root=root)

    def test_recording_output_must_be_under_repo_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outside = root / "recording.wav"

            self.parse_error(["--duration", "3", "--output", str(outside)], "under the repository .artifacts", repo_root=root)

            inside = root / ".artifacts" / "microphone" / "recording.wav"
            args = microphone.parse_args(["--duration", "3", "--output", str(inside)], repo_root=root)

        self.assertEqual(args.output, inside.resolve())
        self.assertEqual(args.duration, 3.0)

    def test_dry_run_does_not_record_or_analyze(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / ".artifacts" / "microphone" / "dry.wav"
            recorder = FakeRecorder()
            lines: list[str] = []
            code = microphone.main(
                ["--dry-run", "--duration", "3", "--output", str(output)],
                recorder_factory=lambda: recorder,
                wav_loader=fail_wav_loader,
                runtime_loader=fail_runtime_loader,
                printer=lines.append,
                repo_root=root,
            )

        self.assertEqual(code, 0)
        self.assertEqual(recorder.calls, [])
        self.assertTrue(any("dry_run=true" in line for line in lines))

    def test_missing_recorder_returns_exit_2_with_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / ".artifacts" / "microphone" / "clip.wav"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = microphone.main(
                    ["--duration", "3", "--output", str(output)],
                    recorder_factory=lambda: None,
                    printer=silent_printer,
                    repo_root=root,
                )

        self.assertEqual(code, 2)
        message = stderr.getvalue()
        self.assertIn("No supported local recorder", message)
        self.assertIn("--input-wav", message)

    def test_recorder_failure_returns_exit_2_with_permission_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / ".artifacts" / "microphone" / "clip.wav"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = microphone.main(
                    ["--duration", "3", "--output", str(output)],
                    recorder_factory=lambda: FakeRecorder(should_fail=True),
                    printer=silent_printer,
                    repo_root=root,
                )

        self.assertEqual(code, 2)
        message = stderr.getvalue()
        self.assertIn("Privacy & Security -> Microphone", message)
        self.assertIn("--input-wav", message)

    def test_existing_wav_mode_with_fake_loader_and_predictor_summarizes_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_wav = root / "clip.wav"
            validate = FakeValidateModule([0.1, 0.5, 0.7, 0.2])
            summary = microphone.run_analysis(
                wav_path=input_wav,
                model_path=root / "model.mlpackage",
                threshold=0.5,
                wav_loader=make_constant_wav_loader([0.0] * (512 * 4 + 7)),
                runtime_loader=make_runtime_loader(validate),
                clock=FakeClock([0.0, 0.001, 0.010, 0.013, 0.020, 0.025, 0.040, 0.042]),
            )

        self.assertEqual(summary.frames, 4)
        self.assertEqual(summary.speech_frames, 2)
        self.assertAlmostEqual(summary.audio_seconds, 0.128)
        self.assertAlmostEqual(summary.speech_seconds, 0.064)
        self.assertEqual(len(summary.segments), 1)
        self.assertAlmostEqual(summary.segments[0].start, 0.032)
        self.assertAlmostEqual(summary.segments[0].end, 0.096)
        self.assertEqual(summary.segments[0].frames, 2)
        self.assertEqual(len(validate.calls), 4)
        self.assertEqual(summary.probability["min"], 0.1)
        self.assertEqual(summary.probability["max"], 0.7)

    def test_json_output_written_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_wav = root / "clip.wav"
            json_output = root / ".artifacts" / "microphone" / "summary.json"
            validate = FakeValidateModule([0.6])
            lines: list[str] = []
            code = microphone.main(
                ["--input-wav", str(input_wav), "--json-output", str(json_output)],
                wav_loader=make_constant_wav_loader([0.0] * 512),
                runtime_loader=make_runtime_loader(validate),
                printer=lines.append,
                repo_root=root,
            )

            payload = parse_json_object(json_output.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["frames"], 1)
        self.assertEqual(payload["speech_frames"], 1)
        segments = cast(list[dict[str, object]], payload["segments"])
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertTrue(any("speech_frames=1" in line for line in lines))

    def test_no_json_file_when_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_wav = root / "clip.wav"
            validate = FakeValidateModule([0.1])
            code = microphone.main(
                ["--input-wav", str(input_wav)],
                wav_loader=make_constant_wav_loader([0.0] * 512),
                runtime_loader=make_runtime_loader(validate),
                printer=silent_printer,
                repo_root=root,
            )

            artifacts = list((root / ".artifacts").glob("**/*")) if (root / ".artifacts").exists() else []

        self.assertEqual(code, 0)
        self.assertEqual(artifacts, [])


class FakeClock:
    _times: list[float]

    def __init__(self, times: list[float]) -> None:
        self._times = list(times)

    def __call__(self) -> float:
        if not self._times:
            raise AssertionError("fake clock exhausted")
        return self._times.pop(0)


if __name__ == "__main__":
    _ = unittest.main()
