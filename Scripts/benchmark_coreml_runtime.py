#!/usr/bin/env python3
"""Plan and run the optional Swift CoreML runtime packaging benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLPACKAGE = REPO_ROOT / "Sources" / "SileroCoreML" / "Resources" / "SileroVAD.mlpackage"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / ".artifacts" / "coreml" / "benchmark"
VALID_MODEL_SUFFIXES = {".mlmodel", ".mlpackage", ".mlmodelc"}
VALID_COMPUTE_UNITS = {"default", "all", "cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine"}
JSONValue: TypeAlias = str | int | bool | None | list[str] | dict[str, "JSONValue"]


class Runner(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        check: bool,
    ) -> object: ...


def subprocess_runner(args: Sequence[str], *, cwd: Path, check: bool) -> object:
    return subprocess.run(list(args), cwd=cwd, check=check)


class BenchmarkPlanError(Exception):
    pass


@dataclass(frozen=True)
class BenchmarkPlan:
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

    def to_jsonable(self, *, dry_run: bool) -> dict[str, JSONValue]:
        return {
            "dry_run": dry_run,
            "mlpackage": str(self.mlpackage),
            "mlmodelc": str(self.mlmodelc) if self.mlmodelc is not None else None,
            "artifacts_dir": str(self.artifacts_dir),
            "compile_command": self.compile_command,
            "benchmark_command": self.benchmark_command,
            "model_paths": [str(path) for path in self.model_paths],
            "chunks": self.chunks,
            "load_repetitions": self.load_repetitions,
            "compute_units": self.compute_units,
            "json_output": str(self.json_output) if self.json_output is not None else None,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or run the Swift SileroVAD runtime benchmark for .mlpackage and .mlmodelc artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--mlpackage",
        default=str(DEFAULT_MLPACKAGE),
        help="Source .mlpackage or .mlmodel artifact to benchmark and optionally compile.",
    )
    _ = parser.add_argument(
        "--mlmodelc",
        default=None,
        help="Existing compiled .mlmodelc artifact to include in the benchmark comparison.",
    )
    _ = parser.add_argument(
        "--compile-if-needed",
        action="store_true",
        help="When --mlmodelc is omitted, compile --mlpackage into --artifacts-dir with xcrun coremlcompiler.",
    )
    _ = parser.add_argument(
        "--artifacts-dir",
        default=str(DEFAULT_ARTIFACTS_DIR),
        help="Ignored local directory for compiled benchmark artifacts.",
    )
    _ = parser.add_argument("--chunks", type=positive_int, default=100, help="Steady-state chunks per model.")
    _ = parser.add_argument(
        "--load-repetitions",
        type=positive_int,
        default=5,
        help="SileroVAD construction/load repetitions per model.",
    )
    _ = parser.add_argument(
        "--compute-units",
        choices=sorted(VALID_COMPUTE_UNITS),
        default="default",
        help="CoreML compute units forwarded to the Swift benchmark.",
    )
    _ = parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON benchmark output path forwarded to the Swift benchmark.",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the JSON plan and execute nothing.",
    )
    return parser


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value}")
    return parsed


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def validate_model_suffix(path: Path, allowed_suffixes: set[str], label: str) -> None:
    if path.suffix.lower() not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise BenchmarkPlanError(f"{label} must end in one of {allowed}: {path}")


def compiled_model_path_for(mlpackage: Path, artifacts_dir: Path) -> Path:
    return artifacts_dir / f"{mlpackage.stem}.mlmodelc"


def build_plan(args: argparse.Namespace) -> BenchmarkPlan:
    mlpackage_arg = cast(str, args.mlpackage)
    mlmodelc_arg = cast(str | None, args.mlmodelc)
    artifacts_dir_arg = cast(str, args.artifacts_dir)
    chunks = cast(int, args.chunks)
    load_repetitions = cast(int, args.load_repetitions)
    compute_units = cast(str, args.compute_units)
    json_output_arg = cast(str | None, args.json_output)
    compile_if_needed = cast(bool, args.compile_if_needed)

    mlpackage = resolve_path(mlpackage_arg)
    artifacts_dir = resolve_path(artifacts_dir_arg)
    json_output = resolve_path(json_output_arg) if json_output_arg else None

    validate_model_suffix(mlpackage, {".mlmodel", ".mlpackage"}, "--mlpackage")

    compile_command: list[str] | None = None
    mlmodelc: Path | None = resolve_path(mlmodelc_arg) if mlmodelc_arg else None
    if mlmodelc is not None:
        validate_model_suffix(mlmodelc, {".mlmodelc"}, "--mlmodelc")
    elif compile_if_needed:
        mlmodelc = compiled_model_path_for(mlpackage, artifacts_dir)
        compile_command = [
            "xcrun",
            "coremlcompiler",
            "compile",
            str(mlpackage),
            str(artifacts_dir),
        ]

    model_paths = [mlpackage]
    if mlmodelc is not None:
        model_paths.append(mlmodelc)
    for model_path in model_paths:
        validate_model_suffix(model_path, VALID_MODEL_SUFFIXES, "model path")

    benchmark_command = ["swift", "run", "SileroVADBenchmark"]
    for model_path in model_paths:
        benchmark_command.extend(["--model-path", str(model_path)])
    benchmark_command.extend(
        [
            "--chunks",
            str(chunks),
            "--load-repetitions",
            str(load_repetitions),
            "--compute-units",
            compute_units,
        ]
    )
    if json_output is not None:
        benchmark_command.extend(["--json-output", str(json_output)])

    return BenchmarkPlan(
        mlpackage=mlpackage,
        mlmodelc=mlmodelc,
        artifacts_dir=artifacts_dir,
        compile_command=compile_command,
        benchmark_command=benchmark_command,
        model_paths=tuple(model_paths),
        chunks=chunks,
        load_repetitions=load_repetitions,
        compute_units=compute_units,
        json_output=json_output,
    )


def run_plan(
    plan: BenchmarkPlan,
    *,
    dry_run: bool,
    runner: Runner = subprocess_runner,
    printer: Callable[[str], None] = print,
) -> int:
    printer(json.dumps(plan.to_jsonable(dry_run=dry_run), indent=2, sort_keys=True))
    if dry_run:
        return 0

    if plan.compile_command is not None:
        plan.artifacts_dir.mkdir(parents=True, exist_ok=True)
        _ = runner(plan.compile_command, cwd=REPO_ROOT, check=True)
    _ = runner(plan.benchmark_command, cwd=REPO_ROOT, check=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        return run_plan(plan, dry_run=cast(bool, args.dry_run))
    except BenchmarkPlanError as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        failed_command = cast(object, exc.cmd)
        print(f"Command failed with exit code {exc.returncode}: {failed_command}", file=sys.stderr)
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
