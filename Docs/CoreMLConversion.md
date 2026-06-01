# CoreML conversion plan

This repository includes a supported local JIT-to-CoreML conversion route, a
Swift runtime facade, and a bundled validated CoreML model. It includes a
preparation/report script, a separate read-only readiness reporter, a
deeper read-only feasibility probe, and an actual converter entrypoint that only
writes a CoreML asset after the manifest-pinned JIT source and conversion
dependencies are available. This document records the recommended Silero VAD to
Core ML conversion workflow.

## Chosen path

Use the official pretrained Silero VAD artifacts, not a retrained model and not
an in-repo upstream checkout.

Recommended workflow:

1. Dry-run acquisition from the pinned manifest.
2. Dry-run conversion preparation/reporting from the same pinned manifest.
3. Read-only conversion readiness reporting from the same pinned manifest.
4. Download official upstream artifacts locally into `.artifacts/`.
5. Read-only conversion feasibility probing against the local artifact.
6. Run the converter entrypoint. It writes a blocked-prerequisites report when
   the local JIT source or conversion dependencies are missing; otherwise it
   writes the CoreML `.mlpackage`.
7. Validate CoreML parity with `Scripts/validate_coreml.py`.
8. Package the validated runtime model with the required MIT notice.
9. Optionally collect local runtime packaging benchmark evidence before any
   future `.mlmodelc` policy discussion.

## Why this path

- It keeps acquisition and conversion reproducible before packaging a runtime artifact.
- It avoids retraining risk when the goal is parity with the official Silero
  behavior.
- It uses the pinned provenance manifest as the source of truth for acquisition.
- It avoids treating an upstream git submodule as a committed dependency.

## Acquisition policy

Use the pinned manifest in `Docs/ModelProvenance/` as the source of truth.

First inspect the planned downloads without writing files:

```bash
python3 Scripts/acquire_silero_artifacts.py --dry-run
```

Then inspect the pinned CoreML conversion target contract without writing files:

```bash
python3 Scripts/prepare_coreml_conversion.py --dry-run
```

The prep script reads the same manifest, defaults to the canonical
`silero_vad.jit` artifact, and prints a JSON plan/report describing:

- upstream release tag and commit
- selected source artifact metadata and SHA-256
- expected local source path under `.artifacts/silero-vad/<upstream_commit>/`
- planned CoreML output path `.artifacts/coreml/SileroVAD.mlpackage`
- the current 16 kHz target contract
- packaging guidance: `.mlpackage` conversion output and bundled SwiftPM runtime resource

This is preparation/reporting only. It does not download upstream assets and it
does not perform CoreML conversion.

Then report conversion readiness without network access or file writes:

```bash
python3 Scripts/report_coreml_conversion_readiness.py
```

The readiness reporter reuses the same pinned manifest and planned paths and
prints a JSON payload that adds:

- dependency availability for local tooling
- planned readiness status for the selected local source artifact
- optional ONNX graph introspection when the selected artifact is ONNX and the
  `onnx` package is already present

That dependency payload now includes local probes for `onnxruntime` too, so the
readiness JSON can show whether ONNX Runtime is available for local smoke,
parity, or probing workflows.

This command is read-only by design. It does not write the planned report path,
does not create directories, does not download assets, and does not convert the
model.

When the selected source is the default JIT artifact, ONNX introspection is
reported as skipped readiness data. If an ONNX artifact is selected but missing,
or if the artifact exists but the `onnx` package is not installed, ONNX
introspection is also reported as skipped readiness data. Those cases do not
make the command fail.

Likewise, missing `onnxruntime` only affects the reported local tooling status.
It does not make readiness reporting fail, because the reporter remains
read-only.

After acquiring artifacts, probe the local conversion route without writing
model outputs or reports by default:

```bash
python3 Scripts/probe_coreml_conversion.py --source-artifact silero_vad.jit
```

The probe reuses the same manifest, path resolution, source selection, and
target contract helpers as the prep/readiness scripts. Its JSON payload adds:

- source artifact existence and SHA-256 verification against the manifest
- dependency checks for `onnx`, `onnxruntime`, `coremltools`, `torch`, and `numpy`
- ONNX interface metadata and ONNX Runtime smoke prediction evidence only when
  the selected source artifact is ONNX; the default JIT source skips these
  ONNX-specific probes
- CoreMLTools route evidence showing whether direct ONNX conversion support is
  observed locally
- the implemented converter route, `torchscript_explicit_state_conversion`

The probe prints JSON to stdout by default. It only writes a JSON report if
`--report path/to/report.json` is explicitly supplied, and it never writes
`.mlpackage`, `.mlmodel`, or `.mlmodelc` outputs.

Missing local artifacts or optional ML dependencies are structured blockers, not
tracebacks. In those cases the command still exits successfully with
`overall_status="blocked"` so the JSON can be consumed by follow-up tooling.

For CoreMLTools 9.0, the probe reports direct ONNX conversion as unsupported
when no local direct ONNX conversion symbol is observed. The converter therefore
uses `torchscript_explicit_state_conversion`: load the manifest-pinned JIT model,
wrap its 16 kHz `stft`, `encoder`, and decoder head modules behind the explicit
`input,state_in -> output,stateN` contract, trace and freeze that wrapper with
`torch.jit.trace` and `torch.jit.freeze`, convert the frozen module with CoreMLTools, then validate
parity before packaging. Do not rely on the current CoreMLTools direct ONNX
route, and do not plan on
`onnx-coreml`; its published repository is obsolete and no longer supported.
This reflects Apple's PyTorch conversion guidance and the CoreMLTools 9.0 release
note facts available during planning: Python 3.13 support, PyTorch 2.7 support,
model state read/write support, and no explicit ONNX conversion support note.

The environment experiments also showed why the recommended conversion setup
should prefer Python 3.13. A Python 3.13 virtual environment with
CoreMLTools 9.0 and PyTorch 2.7 worked cleanly for local tooling. A separate
Python 3.14 environment built CoreMLTools 9.0 from source, selected Torch 2.12
by default, and emitted missing native proxy and Torch untested warnings. Until
those warnings are resolved upstream, prefer the Python 3.13 plus PyTorch 2.7
environment for conversion and validation work.

The actual converter entrypoint is prerequisite-safe:

```bash
python3 Scripts/convert_coreml.py --dry-run
python3 Scripts/convert_coreml.py --source-artifact silero_vad.jit
```

Dry-run mode prints the probe-backed conversion decision JSON, creates no
directories or files, and exits successfully even when prerequisites are missing.
Non-dry-run mode uses the same manifest SHA verification, dependency versions,
target contract, attempted route, and blockers. When the selected JIT artifact,
`torch`, and `coremltools` are available, it writes
`.artifacts/coreml/SileroVAD.mlpackage` and the JSON conversion report to
`.artifacts/coreml/SileroVAD.conversion-report.json`. When prerequisites are
missing, it writes only the report and returns the blocked-prerequisites exit
code. The report still preserves `direct_onnx_conversion` as unsupported
evidence; it is not the intended conversion route.

Then acquire the official artifacts locally:

```bash
python3 Scripts/acquire_silero_artifacts.py --dest .artifacts/silero-vad
```

The downloader stores files under:

```text
.artifacts/silero-vad/<upstream_commit>/
```

Do not commit downloaded upstream artifacts. Commit only the validated converted
runtime resource under `Sources/SileroCoreML/Resources/SileroVAD.mlpackage`.

If runtime packaging measurements are needed after validation, use the optional
benchmark rather than changing the packaged artifact first:

```bash
python3 Scripts/benchmark_coreml_runtime.py --dry-run --compile-if-needed
xcrun coremlcompiler compile \
  Sources/SileroCoreML/Resources/SileroVAD.mlpackage \
  .artifacts/coreml/benchmark
swift run SileroVADBenchmark \
  --model-path Sources/SileroCoreML/Resources/SileroVAD.mlpackage \
  --model-path .artifacts/coreml/benchmark/SileroVAD.mlmodelc
```

The generated `.mlmodelc` and benchmark JSON reports belong under ignored
`.artifacts/` paths. These measurements are evidence for future runtime
packaging policy only; the default packaged artifact remains the validated
`.mlpackage` until a separate decision changes it.

## Upstream checkout policy

Do not add `snakers4/silero-vad` as a committed git submodule.

If future conversion work needs an upstream source checkout for inspection,
place it in the ignored path below and pin it to the manifest commit:

```text
.artifacts/upstream/silero-vad
```

That checkout is for local reference only. It is not the recommended committed
dependency shape for this repository.

## Conversion guidance

The intended source artifact is an official pretrained Silero VAD artifact from
the pinned manifest, not a retrained export built from source.

Use the upstream release tag in the manifest as the provenance source. Do not
pin moving `master` for model generation.

This repository now provides a preparation/report script:

```bash
python3 Scripts/prepare_coreml_conversion.py --dry-run
```

Non-dry-run mode validates that the selected local source artifact exists and
writes a JSON report with `conversion_status="planned"`.

This repository also provides a separate read-only readiness reporter:

```bash
python3 Scripts/report_coreml_conversion_readiness.py
```

It does not perform conversion and does not write the planned readiness report
path. It only reports readiness against the pinned manifest, the planned local
paths, local dependency availability, and optional ONNX metadata when local
inputs make that possible.

Future conversion work should use Apple Core ML Tools and the official PyTorch
conversion workflow as the route reference:

https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html

That workflow is relevant because it recommends converting PyTorch directly by
capturing a graph with `torch.jit.trace` and passing the traced graph to
`ct.convert(...)`, without using ONNX as an intermediate format.

The official quickstart remains useful as a reference:

https://apple.github.io/coremltools/docs-guides/source/introductory-quickstart.html

That quickstart is relevant for conversion work because it covers:

- `ct.convert(...)`
- setting model metadata
- `model.predict(...)` validation on macOS
- `model.save("Name.mlpackage")`

It is not the reference for the downloader or for the runtime packaging
scaffold in this repository.

The supported conversion route uses a TorchScript explicit-state
wrapper that keeps the current runtime contract stable: `input [1,576]`,
`state_in [2,1,128]`, `output`, and `stateN [2,1,128]`. The public input is
named `state_in` because CoreMLTools reserves or rewrites `state` during ML
Program conversion. The released
`silero_vad.jit` artifact is not enough as-is for this v1 contract because its
convenience API owns internal context/state and takes a sampling-rate argument;
the CoreML v1 artifact should expose state as explicit input/output tensors. A
later Core ML stateful model remains optional, but it would require newer OS
targets and should be treated as a separate follow-on design choice rather than
the v1 target.

## Validation requirement

Do not treat a converted model as ready until it passes parity validation.

Install the Python dependencies from `Scripts/requirements.txt` for local
validation and conversion tooling, then run:

```bash
python Scripts/validate_coreml.py --model-path path/to/SileroVAD.mlpackage
```

The current validator expects an explicit-state CoreML interface:

- input `input` with shape `[1, 576]`
- input `state_in` with shape `[2, 1, 128]`
- output `output` as the speech probability
- output `stateN` with shape `[2, 1, 128]`

The validator compares the CoreML model against the official Python Silero VAD
reference implementation. It checks cold-start behavior, streaming behavior,
and reset behavior.

For optional local dataset scoring, this repository also provides
`Scripts/evaluate_ten_vad.py`. It is separate from the parity validator: it runs
a local CoreML model over caller-supplied TEN-VAD `.wav` files, reads the paired
one-line `.scv` annotations, expands labels to the 16 kHz / 512-sample hop used
by the TEN Silero comparison style, and reports threshold-sweep metrics plus
latency/RTFx summaries. The evaluator reuses the existing CoreML loading,
signature, state/context, and prediction helpers from `validate_coreml.py`; it
does not change the validator's random CoreML/PyTorch parity behavior.

TEN fixtures are not vendored and are not required by CI. The default local path,
when present, is:

```text
.artifacts/ten-vad-testset/22a3bcd4509d0faaa8eef4881e8af5f39c178950/testset/
```

The evaluator performs no hidden network access on import, during tests, or when
fixtures are missing. To inspect the explicit pinned acquisition target, run:

```bash
python3 Scripts/evaluate_ten_vad.py --print-fixture-download-plan
```

Then place the 30 `testset-audio-01` through `testset-audio-30` `.wav`/`.scv`
pairs from TEN commit `22a3bcd4509d0faaa8eef4881e8af5f39c178950` under the
ignored `.artifacts/` path or pass another local directory explicitly:

```bash
python3 Scripts/evaluate_ten_vad.py \
  --model-path Sources/SileroCoreML/Resources/SileroVAD.mlpackage \
  --ten-testset-dir .artifacts/ten-vad-testset/22a3bcd4509d0faaa8eef4881e8af5f39c178950/testset \
  --threshold-step 0.01 \
  --json-output .artifacts/ten-vad-results.json
```

Use `--limit-files` for quick local smoke runs and `--thresholds 0.3,0.5,0.7`
when a fixed threshold list is preferred over the default 0.01 sweep. The results
are local CoreML Silero v6.2.1 scores against TEN labels, not a reproduction of
TEN's Silero V5 precision/recall curve.

The validator now owns the CoreML-side host context contract itself. For each
CoreML prediction it concatenates `context[64] + fresh_chunk[512]` into the
`input [1,576]` tensor, passes the explicit recurrent `state_in [2,1,128]`, reads
back `output` and `stateN`, then carries forward the last 64 samples as the next
host context. The PyTorch reference calls remain aligned to upstream behavior by
feeding fresh 512-sample chunks and letting the reference model manage its own
internal streaming state.

Those local Python tooling requirements also cover ONNX-side smoke/probing work.
In particular, `onnxruntime` is tracked as local optional tooling for parity or
inspection workflows when you already have local artifacts, even though no ONNX
Runtime-based conversion path exists in this repository today.

The recommended 16 kHz CoreML runtime contract remains:

- input `input` with shape `[1, 576]`, where `576 = 64` context samples plus
  `512` fresh 16 kHz samples
- input `state_in` with shape `[2, 1, 128]`
- output `output` as the speech probability
- output `stateN` with shape `[2, 1, 128]`

The Swift host wrapper should own the mutable streaming contract:

1. keep `state[2, 1, 128]`
2. keep `context[64]`
3. concatenate `context + raw512`
4. run CoreML inference
5. replace `state` with `stateN`
6. replace `context` with the last 64 samples of the current chunk
7. expose `reset()` to zero state and context

Start with 16 kHz only. Add an 8 kHz model later only after 16 kHz parity is
proven.

## Why 16 kHz first

Silero VAD is small, so model size is not the deciding constraint. The first
CoreML target should optimize for quality, parity, and a simple runtime
contract.

Use 16 kHz first because:

- it is the normal wideband path for app microphone input
- the official workflow uses 512 fresh samples per 16 kHz frame, or about 32 ms
- iOS audio can be resampled into 16 kHz before VAD
- 8 kHz is primarily useful for telephony/narrowband audio
- supporting both sample rates in v1 would add conversion and validation surface
  before the 16 kHz path is proven

Add an 8 kHz model later if Chanora needs telephony-specific behavior or if
device measurements show a reason to specialize.

This repository ships the validated CoreML runtime model as
`Sources/SileroCoreML/Resources/SileroVAD.mlpackage`. Callers may still supply an
explicit model path for parity validation, custom builds, or local conversion
outputs. `.artifacts/` remains an ignored local workspace for upstream downloads,
conversion outputs, and reports.

## Current package truth

Keep the docs honest about the current package state:

- A conversion-prep/report script exists; it plans conversion but does not itself convert.
- A read-only readiness reporter exists, but it is not a conversion command and it does not write reports or assets.
- A read-only feasibility probe exists, but it is not a converter and only writes a JSON report when `--report` is explicitly supplied.
- A prerequisite-safe converter entrypoint exists and writes `.mlpackage` only when the manifest-pinned JIT source and conversion dependencies are available.
- A converted `.mlpackage` runtime resource exists in this repo.
- A runtime inference API exists as `SileroVAD` and supports both the bundled
  default model and caller-supplied validated model URLs.
- No `.mlmodelc` runtime resource exists yet.
