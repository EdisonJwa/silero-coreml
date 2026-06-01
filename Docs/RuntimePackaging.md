# Runtime packaging plan

This repository includes a runtime inference facade and a bundled validated
CoreML model. This document records the current package-resource loading policy
and the external model override path.

## Packaging status

The bundled model was packaged after these conditions were satisfied:

1. The source artifact came from the pinned official manifest.
2. Conversion from the official pretrained artifact is complete.
3. `Scripts/validate_coreml.py` shows acceptable parity.
4. The upstream MIT license notice is included in `THIRD_PARTY_NOTICES.md`.

The package resource is `Sources/SileroCoreML/Resources/SileroVAD.mlpackage`.
Local conversion outputs under `.artifacts/` remain ignored development evidence.

## Current runtime API

The package exposes `SileroVAD` as the narrow runtime facade:

```swift
let vad = try SileroVAD(modelURL: modelURL, configuration: MLModelConfiguration())
let bundledVad = try SileroVAD()
let probability = try vad.process(samples512)
let speech = try vad.isSpeech(samples512)
let coldStartProbability = try vad.processStateless(samples512)
vad.reset()
```

The default initializer loads the bundled `SileroVAD.mlpackage` through
`Bundle.module`. The `modelURL` initializer accepts a caller-supplied file URL to
a validated `.mlpackage`, `.mlmodel`, or `.mlmodelc`. `.mlpackage` and `.mlmodel`
inputs are compiled before Core ML loads them.

`process(_:)` accepts exactly 512 fresh 16 kHz `Float` samples. The Swift host
owns the 64-sample context and the 256-float recurrent state, builds the CoreML
`input` tensor as `context[64] + chunk[512]`, sends `state_in [2,1,128]`, and
carries `stateN` plus the latest context into the next call. `isSpeech` defaults
to `SileroVAD.defaultThreshold` (`0.5`) and uses `>=`. Public runtime constants
are available as `SileroVAD.modelVersion`, `SileroVAD.sampleRate`, and
`SileroVAD.chunkSize`. `processStateless(_:)` uses zero context and zero state
for one call and does not mutate the instance stream. `reset()` is non-throwing
and zeroes context/state.

Use one `SileroVAD` instance per audio stream. The `@unchecked Sendable` facade
owns mutable streaming state and serializes `process`, `processStateless`, and
`reset` calls on a shared instance so stream state updates do not interleave.
Public runtime failures are surfaced as `SileroVADError`, including invalid
chunk size, invalid returned state size, model-contract mismatches, and
underlying CoreML prediction failures.

The runtime uses `MLModel(contentsOf:configuration:)`, `MLMultiArray`, and
`MLDictionaryFeatureProvider` directly after compiling package/model inputs when
needed. It does not depend on generated CoreML model classes.

`SileroVADSegmenter` is pure Swift probability post-processing on top of the
runtime output. It consumes one probability per 512-sample frame, tracks
threshold/minimum-duration/padding state, and emits completed speech segments.
Because it has no CoreML dependency and does not load model resources, it does
not change runtime model packaging, default `Bundle.module` lookup, model
compilation, or the external `modelURL` override path.

## Runtime artifact shape

The runtime package contains one validated model artifact, not a full upstream
source tree.

Current resource:

- a validated `.mlpackage`

The package may switch to a precompiled `.mlmodelc` later if distribution or
startup measurements show that is preferable. Keep `.mlpackage` as the conversion
and audit artifact unless runtime needs force a different choice.

## Optional runtime packaging benchmark

The repository includes an optional local benchmark for measuring `.mlpackage`
and `.mlmodelc` behavior through the same public runtime override path:

```swift
try SileroVAD(modelURL: modelURL, configuration: configuration)
```

The benchmark does not change `SileroVAD()` default lookup, does not add a
packaged `.mlmodelc`, and does not switch runtime packaging policy. It is only
local evidence for a future policy decision.

Inspect the Python wrapper's plan without running Swift, CoreML, or Apple's
compiler:

```bash
python3 Scripts/benchmark_coreml_runtime.py --dry-run --compile-if-needed
```

The default `.mlpackage` input is:

```text
Sources/SileroCoreML/Resources/SileroVAD.mlpackage
```

The default benchmark artifact directory is ignored by git:

```text
.artifacts/coreml/benchmark/
```

To create a local compiled comparison artifact by hand, run:

```bash
xcrun coremlcompiler compile \
  Sources/SileroCoreML/Resources/SileroVAD.mlpackage \
  .artifacts/coreml/benchmark
```

Then compare the bundled `.mlpackage` and generated `.mlmodelc` with the Swift
benchmark executable:

```bash
swift run SileroVADBenchmark \
  --model-path Sources/SileroCoreML/Resources/SileroVAD.mlpackage \
  --model-path .artifacts/coreml/benchmark/SileroVAD.mlmodelc \
  --chunks 100 \
  --load-repetitions 5 \
  --json-output .artifacts/coreml/benchmark/runtime-packaging.json
```

Or let the wrapper compile into `.artifacts/coreml/benchmark/` when no
`--mlmodelc` is supplied:

```bash
python3 Scripts/benchmark_coreml_runtime.py \
  --compile-if-needed \
  --json-output .artifacts/coreml/benchmark/runtime-packaging.json
```

Generated `.mlmodelc` directories and JSON reports stay ignored under
`.artifacts/`. Do not commit them unless a separate packaging policy change is
made and reviewed.

## SwiftPM packaging

The model is distributed as a Swift Package Manager resource and loaded with
`Bundle.module` by `SileroVAD()`.

The package shape is:

- include one validated model resource in the package
- resolve the default model location through `Bundle.module`
- allow an external file URL override for tests and custom builds

The external override matters because it lets tests and local experiments point
at an out-of-repo `.mlpackage` or `.mlmodelc` without changing packaged assets.

The likely v1 runtime wrapper should use explicit state tensors rather than Core
ML's newer stateful model APIs, so the package can support older iOS/macOS
targets. A later iOS 18/macOS 15+ variant can revisit native Core ML state.

## What not to package

Do not package these as the default runtime dependency shape:

- a committed git submodule of `snakers4/silero-vad`
- an upstream source checkout
- unvalidated conversion outputs
- multiple speculative model variants

If an upstream checkout is needed during evaluation, keep it local and ignored
under `.artifacts/upstream/silero-vad` at the pinned commit.

## Current repository truth

At the time of this document:

- a packaged `.mlpackage` exists in this repo
- no packaged `.mlmodelc` exists in this repo
- runtime loading supports both `SileroVAD()` and caller-supplied `modelURL`
- the executable example uses the bundled model by default and accepts an
  optional caller-supplied model path
- a runtime inference API exists as `SileroVAD`
- pure Swift segment post-processing exists as `SileroVADSegmenter`
- tests use an internal fake predictor seam and do not require model assets

This is the current runtime loading and packaging policy document.
