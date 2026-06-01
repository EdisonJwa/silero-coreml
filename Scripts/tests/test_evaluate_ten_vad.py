from __future__ import annotations

import importlib
import json
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class LabelSegmentLike(Protocol):
    def __init__(self, start: float, end: float, label: int) -> None: ...

    start: float
    end: float
    label: int


class TenAnnotationsLike(Protocol):
    def __init__(self, *, filename: str, segments: list[LabelSegmentLike]) -> None: ...

    filename: str
    segments: list[LabelSegmentLike]


class ConfusionCountsLike(Protocol):
    def __init__(self, tp: int = 0, tn: int = 0, fp: int = 0, fn: int = 0) -> None: ...

    tp: int
    tn: int
    fp: int
    fn: int


class FixturePairLike(Protocol):
    stem: str


class EvaluationResultLike(Protocol):
    def __init__(
        self,
        *,
        files: list[object],
        overall: dict[str, float | int],
        sweep: list[dict[str, object]],
        timing: dict[str, float | int],
    ) -> None: ...

    files: list[object]
    overall: dict[str, float | int]
    sweep: list[dict[str, object]]
    timing: dict[str, float | int]


class FileEvaluationLike(Protocol):
    def __init__(
        self,
        *,
        filename: str,
        frames: int,
        label_frames: int,
        prediction_frames: int,
        aligned_frames: int,
        dropped_label_frames: int,
        dropped_prediction_frames: int,
        audio_seconds: float,
        threshold_metrics: dict[str, float | int],
    ) -> None: ...

    filename: str
    frames: int
    label_frames: int
    prediction_frames: int
    aligned_frames: int
    dropped_label_frames: int
    dropped_prediction_frames: int
    audio_seconds: float
    threshold_metrics: dict[str, float | int]


class EvaluateTenVadModule(Protocol):
    EvaluationError: type[Exception]
    TEN_TESTSET_COMMIT: str
    LabelSegment: type[LabelSegmentLike]
    TenAnnotations: type[TenAnnotationsLike]
    ConfusionCounts: type[ConfusionCountsLike]
    FileEvaluation: type[FileEvaluationLike]
    EvaluationResult: type[EvaluationResultLike]

    def parse_scv_line(self, line: str) -> TenAnnotationsLike: ...

    def load_wav_mono_16khz(self, path: Path, *, torchaudio_module: object) -> list[float]: ...

    def expand_labels(
        self,
        segments: list[LabelSegmentLike],
        *,
        hop_size: int,
        sample_rate: int,
    ) -> list[int]: ...

    def chunk_audio_frames(self, samples: list[float], *, chunk_size: int) -> list[list[float]]: ...

    def threshold_probabilities(self, probabilities: list[float], *, threshold: float) -> list[int]: ...

    def confusion_counts(self, labels: list[int], predictions: list[int]) -> ConfusionCountsLike: ...

    def metrics_from_counts(self, counts: ConfusionCountsLike) -> dict[str, float]: ...

    def build_thresholds(self, *, step: float, explicit_thresholds: str | None) -> list[float]: ...

    def threshold_sweep(
        self,
        *,
        labels: list[int],
        probabilities: list[float],
        thresholds: list[float],
    ) -> list[dict[str, object]]: ...

    def time_call(
        self,
        callback: object,
        *,
        clock: FakeClock,
        latencies: list[float],
    ) -> str: ...

    def summarize_latency(self, latencies_seconds: list[float]) -> dict[str, float | int]: ...

    def alignment_accounting(self, label_frames: int, prediction_frames: int) -> dict[str, int]: ...

    def should_warn_alignment_mismatch(self, label_frames: int, prediction_frames: int) -> bool: ...

    def format_alignment_warning(self, filename: str, accounting: dict[str, int]) -> str: ...

    def find_fixture_pairs(self, testset_dir: Path, *, limit_files: int | None) -> list[FixturePairLike]: ...

    def fixture_download_plan(self, artifacts_root: Path) -> dict[str, object]: ...

    def main(self, argv: list[str]) -> int: ...

    def run_evaluation(self, **kwargs: object) -> EvaluationResultLike: ...

    def result_to_jsonable(self, result: EvaluationResultLike) -> dict[str, object]: ...


evaluate = cast(
    EvaluateTenVadModule,
    cast(object, importlib.import_module("evaluate_ten_vad")),
)


class FakeClock:
    def __init__(self, times: list[float]) -> None:
        self._times: list[float] = list(times)

    def __call__(self) -> float:
        if not self._times:
            raise AssertionError("fake clock exhausted")
        return self._times.pop(0)


class EvaluateTenVadTests(unittest.TestCase):
    def write_pcm_wav(
        self,
        path: Path,
        *,
        sample_rate: int,
        channels: int,
        frames: list[tuple[int, ...]],
    ) -> None:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(
                b"".join(struct.pack("<" + "h" * channels, *frame) for frame in frames)
            )

    def test_parse_scv_line_accepts_one_line_triples(self) -> None:
        annotations = evaluate.parse_scv_line(
            "testset-audio-01,0.000,0.403,0,0.403,1.204,1,1.204,1.300,0\n"
        )

        self.assertEqual(annotations.filename, "testset-audio-01")
        self.assertEqual(
            annotations.segments,
            [
                evaluate.LabelSegment(0.0, 0.403, 0),
                evaluate.LabelSegment(0.403, 1.204, 1),
                evaluate.LabelSegment(1.204, 1.3, 0),
            ],
        )

    def test_parse_scv_line_rejects_invalid_triples_labels_and_times(self) -> None:
        with self.assertRaisesRegex(evaluate.EvaluationError, "divisible triples"):
            _ = evaluate.parse_scv_line("file,0.0,0.1,1,0.1")
        with self.assertRaisesRegex(evaluate.EvaluationError, "label must be 0 or 1"):
            _ = evaluate.parse_scv_line("file,0.0,0.1,2")
        with self.assertRaisesRegex(evaluate.EvaluationError, "non-negative"):
            _ = evaluate.parse_scv_line("file,-0.1,0.1,1")
        with self.assertRaisesRegex(evaluate.EvaluationError, "monotonic"):
            _ = evaluate.parse_scv_line("file,0.2,0.1,1")
        with self.assertRaisesRegex(evaluate.EvaluationError, "monotonic"):
            _ = evaluate.parse_scv_line("file,0.0,0.2,1,0.1,0.3,0")

    def test_expand_labels_uses_ten_frame_duration_rounding(self) -> None:
        annotations = evaluate.TenAnnotations(
            filename="clip",
            segments=[
                evaluate.LabelSegment(0.0, 0.016, 0),
                evaluate.LabelSegment(0.016, 0.048, 1),
                evaluate.LabelSegment(0.048, 0.096, 0),
            ],
        )

        labels = evaluate.expand_labels(
            annotations.segments,
            hop_size=512,
            sample_rate=16000,
        )

        self.assertEqual(labels, [1, 0, 0])

    def test_chunk_audio_drops_incomplete_tail(self) -> None:
        chunks = evaluate.chunk_audio_frames(list(range(1025)), chunk_size=512)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[1][0], 512)
        self.assertEqual(chunks[1][-1], 1023)

    def test_inclusive_thresholding_and_confusion_metrics(self) -> None:
        predictions = evaluate.threshold_probabilities([0.1, 0.5, 0.9], threshold=0.5)
        counts = evaluate.confusion_counts(labels=[0, 1, 0], predictions=predictions)
        metrics = evaluate.metrics_from_counts(counts)

        self.assertEqual(predictions, [0, 1, 1])
        self.assertEqual(counts, evaluate.ConfusionCounts(tp=1, tn=1, fp=1, fn=0))
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["fpr"], 0.5)
        self.assertAlmostEqual(metrics["fnr"], 0.0)

    def test_undefined_ratios_return_zero(self) -> None:
        metrics = evaluate.metrics_from_counts(evaluate.ConfusionCounts(tp=0, tn=0, fp=0, fn=0))

        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["fpr"], 0.0)
        self.assertEqual(metrics["fnr"], 0.0)

    def test_threshold_sweep_includes_endpoints(self) -> None:
        thresholds = evaluate.build_thresholds(step=0.25, explicit_thresholds=None)
        sweep = evaluate.threshold_sweep(labels=[0, 1], probabilities=[0.0, 1.0], thresholds=thresholds)

        self.assertEqual(thresholds, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(sweep[0]["threshold"], 0.0)
        self.assertEqual(sweep[-1]["threshold"], 1.0)
        self.assertEqual(sweep[-1]["counts"], {"tp": 1, "tn": 1, "fp": 0, "fn": 0})

    def test_explicit_thresholds_are_sorted_unique_and_validated(self) -> None:
        thresholds = evaluate.build_thresholds(step=0.01, explicit_thresholds="0.7,0,1,0.7")

        self.assertEqual(thresholds, [0.0, 0.7, 1.0])
        with self.assertRaisesRegex(evaluate.EvaluationError, "between 0.0 and 1.0"):
            _ = evaluate.build_thresholds(step=0.01, explicit_thresholds="1.1")

    def test_latency_summary_uses_fake_clock(self) -> None:
        latencies: list[float] = []
        clock = FakeClock([10.0, 10.010, 20.0, 20.020])

        first = evaluate.time_call(lambda: "first", clock=clock, latencies=latencies)
        second = evaluate.time_call(lambda: "second", clock=clock, latencies=latencies)
        summary = evaluate.summarize_latency(latencies)

        self.assertEqual((first, second), ("first", "second"))
        self.assertAlmostEqual(summary["mean_ms"], 15.0)
        self.assertAlmostEqual(summary["p50_ms"], 15.0)
        self.assertAlmostEqual(summary["p95_ms"], 19.5)
        self.assertAlmostEqual(summary["max_ms"], 20.0)

    def test_alignment_accounting_tracks_dropped_frames(self) -> None:
        accounting = evaluate.alignment_accounting(1000, 994)

        self.assertEqual(
            accounting,
            {
                "label_frames": 1000,
                "prediction_frames": 994,
                "aligned_frames": 994,
                "dropped_label_frames": 6,
                "dropped_prediction_frames": 0,
            },
        )

    def test_alignment_warning_threshold_is_deterministic(self) -> None:
        self.assertFalse(evaluate.should_warn_alignment_mismatch(1000, 999))
        self.assertFalse(evaluate.should_warn_alignment_mismatch(1000, 995))
        self.assertTrue(evaluate.should_warn_alignment_mismatch(1000, 994))

        warning = evaluate.format_alignment_warning(
            "clip",
            evaluate.alignment_accounting(1000, 994),
        )

        self.assertIn("warning: file=clip alignment mismatch", warning)
        self.assertIn("mismatch_frames=6", warning)
        self.assertIn("dropped_label_frames=6", warning)
        self.assertIn("dropped_prediction_frames=0", warning)

    def test_find_fixture_pairs_and_default_missing_message_do_not_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            testset = root / "testset"
            testset.mkdir()
            _ = (testset / "testset-audio-01.wav").write_bytes(b"fake")
            _ = (testset / "testset-audio-01.scv").write_text(
                "testset-audio-01,0,0.032,0\n",
                encoding="utf-8",
            )

            pairs = evaluate.find_fixture_pairs(testset, limit_files=None)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].stem, "testset-audio-01")
            with self.assertRaisesRegex(evaluate.EvaluationError, "TEN fixtures are missing"):
                _ = evaluate.find_fixture_pairs(root / "missing", limit_files=None)

    def test_load_wav_mono_16khz_falls_back_for_backendless_mono_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "mono.wav"
            self.write_pcm_wav(
                wav_path,
                sample_rate=16000,
                channels=1,
                frames=[(-32768,), (0,), (32767,)],
            )

            torchaudio_stub = SimpleNamespace(
                load=mock.Mock(
                    side_effect=RuntimeError(
                        "Couldn't find appropriate backend to handle uri mono.wav and format None."
                    )
                )
            )

            samples = evaluate.load_wav_mono_16khz(wav_path, torchaudio_module=torchaudio_stub)

        self.assertEqual(len(samples), 3)
        self.assertAlmostEqual(samples[0], -1.0)
        self.assertAlmostEqual(samples[1], 0.0)
        self.assertAlmostEqual(samples[2], 32767 / 32768)

    def test_load_wav_mono_16khz_falls_back_and_averages_stereo_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "stereo.wav"
            self.write_pcm_wav(
                wav_path,
                sample_rate=16000,
                channels=2,
                frames=[(-32768, 32767), (16384, 16384)],
            )

            torchaudio_stub = SimpleNamespace(
                load=mock.Mock(
                    side_effect=RuntimeError(
                        "Couldn't find appropriate backend to handle uri stereo.wav and format None."
                    )
                )
            )

            samples = evaluate.load_wav_mono_16khz(wav_path, torchaudio_module=torchaudio_stub)

        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[0], (-1.0 + (32767 / 32768)) / 2)
        self.assertAlmostEqual(samples[1], 0.5)

    def test_load_wav_mono_16khz_fallback_rejects_unsupported_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "wrong-rate.wav"
            self.write_pcm_wav(
                wav_path,
                sample_rate=8000,
                channels=1,
                frames=[(0,), (1,)],
            )

            torchaudio_stub = SimpleNamespace(
                load=mock.Mock(
                    side_effect=RuntimeError(
                        "Couldn't find appropriate backend to handle uri wrong-rate.wav and format None."
                    )
                )
            )

            with self.assertRaisesRegex(evaluate.EvaluationError, "only supports 16000 Hz PCM WAV"):
                _ = evaluate.load_wav_mono_16khz(wav_path, torchaudio_module=torchaudio_stub)

    def test_print_download_plan_is_explicit_and_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = evaluate.fixture_download_plan(Path(temp_dir))

        self.assertEqual(plan["commit"], evaluate.TEN_TESTSET_COMMIT)
        self.assertTrue(str(plan["destination"]).endswith("testset"))
        self.assertIn("no network access", cast(str, plan["note"]))

    def test_run_evaluation_warns_and_records_alignment_accounting(self) -> None:
        fake_pair = SimpleNamespace(
            stem="testset-audio-01",
            wav_path=Path("fake.wav"),
            scv_path=Path("fake.scv"),
        )
        fake_annotations = evaluate.TenAnnotations(filename="testset-audio-01", segments=[])
        fake_file_metrics = {
            "threshold": 0.5,
            "tp": 1,
            "tn": 1,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "accuracy": 1.0,
            "fpr": 0.0,
            "fnr": 0.0,
        }
        fake_sweep = [{"threshold": 0.5, "counts": {"tp": 1, "tn": 1, "fp": 0, "fn": 0}}]
        printed: list[str] = []

        with (
            mock.patch.object(evaluate, "find_fixture_pairs", return_value=[fake_pair]),
            mock.patch.object(evaluate, "load_torchaudio_module", return_value=object()),
            mock.patch.object(evaluate, "load_coreml_runtime", return_value=(object(), object(), object())),
            mock.patch.object(evaluate, "parse_scv_file", return_value=fake_annotations),
            mock.patch.object(evaluate, "load_wav_mono_16khz", return_value=[0.0]),
            mock.patch.object(evaluate, "chunk_audio_frames", return_value=[[0.0]] * 994),
            mock.patch.object(evaluate, "expand_labels", return_value=[1] * 1000),
            mock.patch.object(
                evaluate,
                "predict_file_probabilities",
                return_value=([0.9] * 994, [0.001] * 994),
            ),
            mock.patch.object(evaluate, "threshold_sweep", return_value=fake_sweep),
            mock.patch.object(evaluate, "select_overall_threshold", return_value=fake_file_metrics),
        ):
            result = evaluate.run_evaluation(
                model_path=Path("fake.mlpackage"),
                testset_dir=Path("fake-testset"),
                thresholds=[0.5],
                printer=printed.append,
            )

        self.assertEqual(len(result.files), 1)
        file_result = cast(FileEvaluationLike, result.files[0])
        self.assertEqual(file_result.filename, "testset-audio-01")
        self.assertEqual(file_result.frames, 994)
        self.assertEqual(file_result.label_frames, 1000)
        self.assertEqual(file_result.prediction_frames, 994)
        self.assertEqual(file_result.aligned_frames, 994)
        self.assertEqual(file_result.dropped_label_frames, 6)
        self.assertEqual(file_result.dropped_prediction_frames, 0)
        self.assertTrue(printed[0].startswith("warning: file=testset-audio-01 alignment mismatch"))
        self.assertIn("label_frames=1000", printed[0])
        self.assertIn("prediction_frames=994", printed[0])
        self.assertIn("aligned_frames=994", printed[1])
        self.assertIn("overall frames=994", printed[2])

        payload = evaluate.result_to_jsonable(result)
        file_payload = cast(list[dict[str, object]], payload["files"])[0]
        self.assertEqual(file_payload["label_frames"], 1000)
        self.assertEqual(file_payload["prediction_frames"], 994)
        self.assertEqual(file_payload["aligned_frames"], 994)
        self.assertEqual(file_payload["dropped_label_frames"], 6)
        self.assertEqual(file_payload["dropped_prediction_frames"], 0)

    def test_main_writes_json_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.mlpackage"
            model_path.mkdir()
            testset = root / "testset"
            testset.mkdir()
            json_output = root / "report.json"

            fake_result = evaluate.EvaluationResult(
                files=[
                    evaluate.FileEvaluation(
                        filename="testset-audio-01",
                        frames=10,
                        label_frames=11,
                        prediction_frames=10,
                        aligned_frames=10,
                        dropped_label_frames=1,
                        dropped_prediction_frames=0,
                        audio_seconds=0.32,
                        threshold_metrics={"threshold": 0.5, "precision": 0.0},
                    )
                ],
                overall={"threshold": 0.5, "precision": 0.0},
                sweep=[],
                timing={"inference_seconds": 0.0},
            )

            with mock.patch.object(evaluate, "run_evaluation", return_value=fake_result):
                exit_code = evaluate.main(
                    [
                        "--model-path",
                        str(model_path),
                        "--ten-testset-dir",
                        str(testset),
                        "--json-output",
                        str(json_output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = cast(dict[str, object], json.loads(json_output.read_text(encoding="utf-8")))
            files = cast(list[dict[str, object]], payload["files"])
            overall = cast(dict[str, object], payload["overall"])
            self.assertEqual(files[0]["label_frames"], 11)
            self.assertEqual(files[0]["prediction_frames"], 10)
            self.assertEqual(files[0]["aligned_frames"], 10)
            self.assertEqual(files[0]["dropped_label_frames"], 1)
            self.assertEqual(files[0]["dropped_prediction_frames"], 0)
            self.assertEqual(overall["threshold"], 0.5)


if __name__ == "__main__":
    _ = unittest.main()
