from __future__ import annotations

import contextlib
import argparse
import importlib
import io
import json
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class BenchmarkPlanLike(Protocol):
    mlpackage: Path
    mlmodelc: Path | None
    artifacts_dir: Path
    compile_command: list[str] | None
    benchmark_command: list[str]
    model_paths: tuple[Path, ...]
    chunks: int
    load_repetitions: int
    compute_units: str
    json_output: Path | None


class RunnerLike(Protocol):
    def __call__(self, command: Sequence[str], *, cwd: Path, check: bool) -> object: ...


class BenchmarkCoreMLRuntimeModule(Protocol):
    REPO_ROOT: Path
    BenchmarkPlanError: type[Exception]

    def build_parser(self) -> argparse.ArgumentParser: ...

    def build_plan(self, args: argparse.Namespace) -> BenchmarkPlanLike: ...

    def run_plan(
        self,
        plan: BenchmarkPlanLike,
        *,
        dry_run: bool,
        runner: RunnerLike,
        printer: PrinterLike,
    ) -> int: ...

    def main(self, argv: Sequence[str] | None = None) -> int: ...


class PrinterLike(Protocol):
    def __call__(self, line: str) -> None: ...


benchmark = cast(
    BenchmarkCoreMLRuntimeModule,
    cast(object, importlib.import_module("benchmark_coreml_runtime")),
)


def parse_json_object(text: str) -> dict[str, object]:
    payload = cast(object, json.loads(text))
    if not isinstance(payload, dict):
        raise AssertionError("expected JSON object")
    return cast(dict[str, object], payload)


def require_str(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise AssertionError(f"expected {key} to be a string")
    return value


def require_optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise AssertionError(f"expected {key} to be a string or null")
    return value


def require_bool(payload: dict[str, object], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise AssertionError(f"expected {key} to be a boolean")
    return value


def require_int(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int):
        raise AssertionError(f"expected {key} to be an integer")
    return value


def require_str_list(payload: dict[str, object], key: str) -> list[str]:
    value = payload[key]
    if not isinstance(value, list):
        raise AssertionError(f"expected {key} to be a list of strings")
    items = cast(list[object], value)
    strings: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise AssertionError(f"expected {key} to be a list of strings")
        strings.append(item)
    return strings


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command: Sequence[str], *, cwd: Path, check: bool) -> object:
        _ = cwd
        _ = check
        self.calls.append(list(command))
        return None


class BufferingPrinter:
    def __init__(self, buffer: io.StringIO) -> None:
        self._buffer: io.StringIO = buffer

    def __call__(self, line: str) -> None:
        _ = self._buffer.write(line)


def discard_line(line: str) -> None:
    _ = line
    return None


class BenchmarkCoreMLRuntimeTests(unittest.TestCase):
    def test_default_dry_run_compile_plan_executes_nothing(self) -> None:
        stdout = io.StringIO()
        runner = RecordingRunner()

        exit_code = benchmark.run_plan(
            benchmark.build_plan(
                benchmark.build_parser().parse_args(
                    [
                        "--compile-if-needed",
                        "--dry-run",
                    ]
                )
            ),
            dry_run=True,
            runner=runner,
            printer=BufferingPrinter(stdout),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(runner.calls, [])
        payload = parse_json_object(stdout.getvalue())
        self.assertTrue(require_bool(payload, "dry_run"))
        mlpackage = require_str(payload, "mlpackage")
        artifacts_dir = require_str(payload, "artifacts_dir")
        compile_command = require_str_list(payload, "compile_command")
        mlmodelc = require_optional_str(payload, "mlmodelc")
        benchmark_command = require_str_list(payload, "benchmark_command")

        self.assertIsNotNone(mlmodelc)
        assert mlmodelc is not None
        self.assertEqual(
            mlpackage,
            str((benchmark.REPO_ROOT / "Sources" / "SileroCoreML" / "Resources" / "SileroVAD.mlpackage").resolve()),
        )
        self.assertEqual(
            artifacts_dir,
            str((benchmark.REPO_ROOT / ".artifacts" / "coreml" / "benchmark").resolve()),
        )
        self.assertEqual(
            compile_command,
            [
                "xcrun",
                "coremlcompiler",
                "compile",
                mlpackage,
                artifacts_dir,
            ],
        )
        self.assertEqual(mlmodelc, str(Path(artifacts_dir) / "SileroVAD.mlmodelc"))
        self.assertIn("SileroVADBenchmark", benchmark_command)
        self.assertIn("--model-path", benchmark_command)
        self.assertIn(mlpackage, benchmark_command)
        self.assertIn(mlmodelc, benchmark_command)

    def test_existing_mlmodelc_skips_compile_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = benchmark.build_parser().parse_args(
                [
                    "--mlpackage",
                    str(root / "SileroVAD.mlpackage"),
                    "--mlmodelc",
                    str(root / "SileroVAD.mlmodelc"),
                    "--chunks",
                    "3",
                    "--load-repetitions",
                    "1",
                    "--compute-units",
                    "cpuOnly",
                    "--json-output",
                    str(root / "result.json"),
                ]
            )

            plan = benchmark.build_plan(args)

            self.assertIsNone(plan.compile_command)
            self.assertEqual(plan.chunks, 3)
            self.assertEqual(plan.load_repetitions, 1)
            self.assertEqual(plan.compute_units, "cpuOnly")
            self.assertEqual(plan.json_output, (root / "result.json").resolve())
            self.assertEqual(plan.model_paths, ((root / "SileroVAD.mlpackage").resolve(), (root / "SileroVAD.mlmodelc").resolve()))

    def test_compile_if_needed_uses_input_model_stem_for_compiled_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = benchmark.build_parser().parse_args(
                [
                    "--mlpackage",
                    str(root / "Foo.mlpackage"),
                    "--compile-if-needed",
                    "--artifacts-dir",
                    str(root / "artifacts"),
                ]
            )

            plan = benchmark.build_plan(args)

            self.assertEqual(plan.mlmodelc, (root / "artifacts" / "Foo.mlmodelc").resolve())
            self.assertEqual(
                plan.compile_command,
                [
                    "xcrun",
                    "coremlcompiler",
                    "compile",
                    str((root / "Foo.mlpackage").resolve()),
                    str((root / "artifacts").resolve()),
                ],
            )
            self.assertEqual(
                plan.model_paths,
                ((root / "Foo.mlpackage").resolve(), (root / "artifacts" / "Foo.mlmodelc").resolve()),
            )

    def test_non_dry_run_uses_subprocess_argument_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = benchmark.build_parser().parse_args(
                [
                    "--mlpackage",
                    str(root / "SileroVAD.mlpackage"),
                    "--compile-if-needed",
                    "--artifacts-dir",
                    str(root / "artifacts"),
                ]
            )
            plan = benchmark.build_plan(args)
            runner = RecordingRunner()

            exit_code = benchmark.run_plan(
                plan,
                dry_run=False,
                runner=runner,
                printer=discard_line,
            )

            self.assertIsNotNone(plan.compile_command)
            assert plan.compile_command is not None
            self.assertEqual(exit_code, 0)
            self.assertEqual(runner.calls, [plan.compile_command, plan.benchmark_command])
            self.assertTrue((root / "artifacts").exists())

    def test_rejects_invalid_suffixes(self) -> None:
        parser = benchmark.build_parser()
        args = parser.parse_args(["--mlpackage", "SileroVAD.txt"])

        with self.assertRaisesRegex(benchmark.BenchmarkPlanError, "--mlpackage must end"):
            _ = benchmark.build_plan(args)

        args = parser.parse_args(["--mlpackage", "SileroVAD.mlpackage", "--mlmodelc", "SileroVAD.mlpackage"])
        with self.assertRaisesRegex(benchmark.BenchmarkPlanError, "--mlmodelc must end"):
            _ = benchmark.build_plan(args)

    def test_main_dry_run_prints_json_without_running_commands(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = benchmark.main(["--dry-run", "--chunks", "2", "--load-repetitions", "1"])

        self.assertEqual(exit_code, 0)
        payload = parse_json_object(stdout.getvalue())
        self.assertTrue(require_bool(payload, "dry_run"))
        self.assertIsNone(payload["compile_command"])
        self.assertEqual(require_int(payload, "chunks"), 2)
        self.assertEqual(require_int(payload, "load_repetitions"), 1)
        self.assertEqual(require_str_list(payload, "model_paths"), [require_str(payload, "mlpackage")])


if __name__ == "__main__":
    _ = unittest.main()
