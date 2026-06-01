# SileroCoreML

Private Chanora-owned Apple/CoreML Silero VAD backend scaffold for eventual `VadBackend::SileroCoreMl` integration.

## Scope of this package

- Establish Swift Package Manager package identity for `SileroCoreML`.
- Reserve a minimal backend identifier surface for Chanora integration.
- Expose a small `SileroVAD` runtime facade backed by a bundled validated CoreML model.
- Provide pure Swift probability post-processing with `SileroVADSegmenter` for completed speech segments.
- Preserve an explicit `modelURL` override for custom builds, tests, and local experiments.
- Record upstream provenance and MIT attribution for the bundled Silero VAD model.

## Runtime API

The package exposes `SileroVAD`, a SileroVADKit-style wrapper around the current
explicit-state CoreML contract:

```swift
import CoreML
import SileroCoreML

let configuration = MLModelConfiguration()
let vad = try SileroVAD(configuration: configuration)
let customVad = try SileroVAD(modelURL: modelURL, configuration: configuration)

let probability = try vad.process(samples512)
let speech = try vad.isSpeech(samples512, threshold: 0.5)
let coldStartProbability = try vad.processStateless(samples512)
vad.reset()
```

Callers pass exactly 512 fresh 16 kHz `Float` samples to `process(_:)` or
`isSpeech(_:threshold:)`. The wrapper owns the streaming state: it prepends the
previous 64-sample context, sends `input [1,576]` and `state_in [2,1,128]` to
CoreML, reads `output` and `stateN`, then carries the returned state and latest
context into the next call. `isSpeech` uses an inclusive `>=` comparison and
defaults to `SileroVAD.defaultThreshold` (`0.5`). Public runtime constants are
available as `SileroVAD.modelVersion`, `SileroVAD.sampleRate`, and
`SileroVAD.chunkSize`. `processStateless(_:)` performs a cold-start call with
zero context and zero state without mutating the instance stream.

The package bundles the validated explicit-state CoreML model at
`Sources/SileroCoreML/Resources/SileroVAD.mlpackage`. The default initializer
loads that resource with `Bundle.module`. The `modelURL` initializer remains
available for tests, custom builds, and local experiments with another validated
`.mlpackage`, `.mlmodel`, or `.mlmodelc`. Runtime loading compiles `.mlpackage`
and `.mlmodel` inputs before loading them with Core ML.
Use one `SileroVAD` instance per audio stream. The `@unchecked Sendable` facade
serializes access to its mutable stream state and CoreML predictor so
shared-instance calls do not interleave.
Public failures are reported as `SileroVADError`, including invalid chunk size,
invalid returned state size, model-contract mismatches, and underlying CoreML
prediction failures.

`SileroVADSegmenter` is a pure Swift post-processing layer for one probability
per 512-sample frame. It applies entry/exit thresholds, minimum speech/silence
durations, and optional speech padding to emit completed `SileroVADSegment`
values with inclusive start frame indexes and exclusive end frame indexes. It
does not load CoreML models and does not consume audio samples directly.
`SileroVADSegmenterConfiguration` requires finite thresholds with
`entryThreshold >= exitThreshold`, and requires `minSpeechDuration`,
`minSilenceDuration`, and `speechPadding` to be finite and non-negative.

```swift
var segmenter = SileroVADSegmenter(
    configuration: SileroVADSegmenterConfiguration(
        entryThreshold: 0.5,
        exitThreshold: 0.35,
        minSpeechDuration: 0.25,
        minSilenceDuration: 0.10,
        speechPadding: 0.03
    )
)

for samples512 in audioFrames {
    let probability = try vad.process(samples512)
    let completedSegments = segmenter.process(probability: probability)
    for segment in completedSegments {
        print("speech from \(segment.startTime) to \(segment.endTime)")
    }
}

let trailingSegments = segmenter.finalize()
```

## Example

The `SileroVADExample` executable runs one 512-sample silent chunk through the
bundled model by default:

```bash
swift run SileroVADExample
```

You can also provide an explicit model path to test another validated CoreML
artifact:

```bash
swift run SileroVADExample .artifacts/coreml/SileroVAD.mlpackage
```

## Validation

Run the package tests with:

```bash
swift test
```

Run the Python unit tests with:

```bash
python3 -m unittest discover -s Scripts/tests
```

To verify the pinned upstream Silero VAD acquisition plan without downloading
any assets, run:

```bash
python3 Scripts/acquire_silero_artifacts.py --dry-run
```

This reads `Docs/ModelProvenance/silero-vad-v6.2.1.json` and prints the
planned official `snakers4/silero-vad` downloads into the ignored
`.artifacts/silero-vad/<upstream_commit>/` directory.

To inspect the pinned CoreML conversion target contract without downloading,
creating directories, or converting anything, run:

```bash
python3 Scripts/prepare_coreml_conversion.py --dry-run
```

This reads the same pinned manifest, selects `silero_vad.jit` by default, and
prints a JSON conversion plan/report for the current 16 kHz CoreML target:
input `[1,576]`, state input `state_in [2,1,128]`, host-managed context `64`, chunk size
`512`, probability output, and next-state output. It also records the planned
`.mlpackage` conversion output path and the future `.mlmodelc` SwiftPM runtime
resource recommendation. The script does not download assets and does not run
CoreML conversion yet.

To report CoreML conversion readiness without network access, file writes, or
asset generation, run:

```bash
python3 Scripts/report_coreml_conversion_readiness.py
```

This command reads the same pinned manifest and planned paths, reports local
dependency availability, and only inspects the ONNX graph when the selected
source artifact is ONNX and the `onnx` package is already present. Missing
artifacts or optional dependencies are reported as skipped readiness data, not
as command failures. The dependency payload includes local tooling probes for
packages such as `onnx`, `onnxruntime`, `coremltools`, `torch`, `numpy`, and
`silero-vad`, but the reporter still succeeds when those optional packages are
absent. This is readiness reporting only. It does not convert, write a report,
create assets, or change the ignored `.artifacts/` workspace.

To probe the local conversion evidence more deeply after acquiring artifacts,
run:

```bash
python3 Scripts/probe_coreml_conversion.py --source-artifact silero_vad.jit
```

The probe prints JSON to stdout by default and remains read-only unless
`--report path/to/report.json` is supplied. It verifies the local source
artifact SHA-256 against the manifest, skips ONNX-specific probes for the JIT
source, and records direct ONNX as unsupported evidence when the installed
`coremltools` exposes no direct ONNX route. Missing artifacts or ML dependencies
are reported as structured blockers with `overall_status="blocked"`; they should
not produce tracebacks or model output files.

To invoke the actual JIT-to-CoreML converter entrypoint, run:

```bash
python3 Scripts/convert_coreml.py --dry-run
```

Dry-run mode prints the conversion decision JSON and creates no files or
directories. Non-dry-run mode verifies the manifest-pinned JIT source and the
conversion dependencies, traces an explicit-state PyTorch wrapper with
`torch.jit.trace`, converts that graph with CoreMLTools, writes
`.artifacts/coreml/SileroVAD.mlpackage`, and writes
`.artifacts/coreml/SileroVAD.conversion-report.json`. If artifacts or conversion
dependencies are missing, it writes only the report and returns the blocked
prerequisites exit code. Direct ONNX remains in the report as unsupported probe
evidence, not as the intended conversion route.

For CoreML parity validation, install the Python dependencies from
`Scripts/requirements.txt` and run:

```bash
python Scripts/validate_coreml.py --model-path path/to/SileroVAD.mlpackage
```

The package includes the bundled runtime model. You can still supply an explicit
CoreML model path when validating local conversion outputs or custom artifacts.

`Scripts/validate_coreml.py` remains the random CoreML/PyTorch parity validator:
it compares deterministic synthetic 512-sample streams against the official
Silero VAD v6.2.1 Python model and does not use TEN-VAD fixtures.

For optional local TEN-VAD fixture evaluation, first inspect the pinned fixture
plan without network access:

```bash
python3 Scripts/evaluate_ten_vad.py --print-fixture-download-plan
```

The evaluator expects the TEN testset from commit
`22a3bcd4509d0faaa8eef4881e8af5f39c178950` to be supplied explicitly as local
`.wav`/`.scv` pairs. TEN audio and label files are not vendored, are not run by
CI, and should stay under the ignored `.artifacts/` workspace, for example:

```text
.artifacts/ten-vad-testset/22a3bcd4509d0faaa8eef4881e8af5f39c178950/testset/
```

Once local fixtures and the optional Python dependencies are available, run:

```bash
python3 Scripts/evaluate_ten_vad.py \
  --model-path Sources/SileroCoreML/Resources/SileroVAD.mlpackage \
  --ten-testset-dir .artifacts/ten-vad-testset/22a3bcd4509d0faaa8eef4881e8af5f39c178950/testset \
  --json-output .artifacts/ten-vad-results.json
```

Use `python3 Scripts/evaluate_ten_vad.py --help` for the full CLI, including
`--threshold-step`, `--thresholds`, and `--limit-files`. The reported metrics are
for the local CoreML Silero v6.2.1 model against TEN labels using 512-sample
windows; they are not a reproduction of TEN's Silero V5 PR curve.

For optional microphone or local WAV smoke validation of the CoreML VAD, inspect
the dry-run plan first:

```bash
python3 Scripts/validate_microphone_vad.py \
  --dry-run \
  --duration 3 \
  --output .artifacts/microphone/mic-smoke.wav \
  --model-path Sources/SileroCoreML/Resources/SileroVAD.mlpackage
```

The script never records unless both `--duration` and an `.artifacts/` `--output`
are supplied. It can also analyze an existing uncompressed 16 kHz PCM WAV with
`--input-wav`. Generated recordings and JSON summaries should stay under the
ignored `.artifacts/` workspace and remain uncommitted. See
`Docs/MicrophoneValidation.md` for safe usage, dependency notes, and microphone
permission troubleshooting.

To smoke-test the Swift facade against a local ignored model artifact, set
`SILERO_COREML_MODEL_PATH` when running Swift tests:

```bash
SILERO_COREML_MODEL_PATH=.artifacts/coreml/SileroVAD.mlpackage swift test
```

To collect optional local runtime packaging evidence, first inspect the planned
benchmark commands without compiling or running CoreML:

```bash
python3 Scripts/benchmark_coreml_runtime.py --dry-run --compile-if-needed
```

The wrapper defaults to the bundled
`Sources/SileroCoreML/Resources/SileroVAD.mlpackage`, plans generated artifacts
under the ignored `.artifacts/coreml/benchmark/` directory, and invokes the
Swift `SileroVADBenchmark` executable with subprocess argument arrays. When you
want to create a local compiled artifact explicitly, run Apple's compiler into
that ignored workspace:

```bash
xcrun coremlcompiler compile \
  Sources/SileroCoreML/Resources/SileroVAD.mlpackage \
  .artifacts/coreml/benchmark
```

Then compare the bundled `.mlpackage` against the generated `.mlmodelc` through
the same `SileroVAD(modelURL:configuration:)` runtime path:

```bash
swift run SileroVADBenchmark \
  --model-path Sources/SileroCoreML/Resources/SileroVAD.mlpackage \
  --model-path .artifacts/coreml/benchmark/SileroVAD.mlmodelc \
  --chunks 100 \
  --load-repetitions 5 \
  --json-output .artifacts/coreml/benchmark/runtime-packaging.json
```

Generated `.mlmodelc` directories and JSON benchmark outputs should stay under
`.artifacts/` and remain uncommitted. This benchmark is evidence for a future
runtime packaging decision; it does not switch the default bundled artifact or
the current `.mlpackage` packaging policy.

The validator now targets the final explicit-state CoreML contract used by the
current planning and probing scripts: input `input [1,576]`, state
`state_in [2,1,128]`, output `output`, and next state `stateN [2,1,128]`. On the
CoreML side, the host owns the 64-sample context and concatenates
`context[64] + fresh_chunk[512]` into the model input. The PyTorch reference
calls still run on fresh 512-sample chunks.

`Scripts/requirements.txt` is the local tooling bundle for conversion,
parity validation, ONNX smoke/probing work, and optional TEN/audio evaluation.
It now includes `onnxruntime` alongside `onnx`, `coremltools`, and
`torchaudio`, but installing those tools is still optional for the read-only
readiness reporter and local evaluator workflows.

For future conversion or parity work, prefer the clean Python 3.13 environment
used during the CoreMLTools 9.0 experiments. CoreMLTools 9.0 release notes
advertise Python 3.13 support, PyTorch 2.7 support, and model state read/write
support. By contrast, the Python 3.14 virtual environment built CoreMLTools
9.0 from source, selected untested Torch 2.12 by default, and emitted missing
native proxy plus Torch untested warnings. The same experiments did not find an
explicit CoreMLTools 9.0 release note promising ONNX conversion support, so the
converter uses the implemented `torchscript_explicit_state_conversion` route
instead. It loads the official `silero_vad.jit` artifact, wraps the 16 kHz
internal `stft`, `encoder`, and decoder head modules behind the final
`input,state_in -> output,stateN` contract, traces and freezes that wrapper with
`torch.jit.trace` and `torch.jit.freeze`, and converts the frozen module with CoreMLTools. Keep the
explicit-state CoreML artifact as the preferred v1 runtime shape; a later Core
ML stateful model variant remains optional and would change the runtime API.

## Documentation

- `Docs/CoreMLConversion.md` explains the recommended Silero VAD to Core ML
  path: pinned manifest, local acquisition into `.artifacts/`, conversion prep
  reporting, read-only readiness/probe reporting, JIT conversion through an
  explicit-state PyTorch wrapper, parity validation, then possible later
  packaging.
- `Docs/RuntimePackaging.md` explains the current runtime packaging shape:
  the bundled `.mlpackage` runtime resource, the `Bundle.module` default
  initializer, and the explicit `modelURL` override for tests and custom builds.
- `Docs/MicrophoneValidation.md` explains optional local WAV/microphone VAD
  validation, including dry-run safety, `.artifacts/` recording policy, and
  macOS microphone permission troubleshooting.

For future conversion work, Apple Core ML Tools quickstart is the main reference
for `ct.convert(...)`, metadata, predict-based validation on macOS, and
`model.save("Name.mlpackage")`:

https://apple.github.io/coremltools/docs-guides/source/introductory-quickstart.html

Apple's PyTorch conversion workflow is the main route reference for the
JIT/TorchScript path. It recommends capturing a PyTorch graph with
`torch.jit.trace` and converting that graph directly with `ct.convert(...)`,
without using ONNX as an intermediate format:

https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html

## Explicit non-goals in this repository state

- No upstream `SileroVADKit` code is included.
- No generated CoreML wrappers or generated model classes are included.
- No runtime audio capture, resampling, threading, or Rust-bridge APIs are
  implemented yet.

## Intended direction

This package is expected to become a distinct Apple/CoreML VAD backend used by Chanora while preserving Chanora's Rust-owned VAD gate policy. The current package defines the CoreML conversion workflow plus the narrow Swift runtime facade, bundled validated model, and pure Swift probability segmenter; higher-level audio capture, resampling, and Rust bridge integration remain outside this repository state.
