# Microphone VAD validation

`Scripts/validate_microphone_vad.py` is an optional local smoke-test for the
bundled explicit-state CoreML Silero VAD model. It can analyze an existing WAV
file, or it can record a short microphone clip only when recording mode is
requested explicitly.

## Safety policy

- `--help` and `--dry-run` do not import CoreML, numpy, coremltools, touch the
  microphone, load WAV data, or run inference.
- There is no hidden recording path. The script records only when both
  `--duration SECONDS` and `--output .artifacts/microphone/name.wav` are present.
- Recording output must be under this repository's ignored `.artifacts/`
  directory. Do not commit recordings or generated JSON summaries.
- Tests use fake recorders, fake WAV loaders, and fake predictors. They do not
  require a microphone, CoreML, recorder command, network access, or optional
  Python dependencies.

## Inspect the plan first

Use dry-run mode before recording. This prints the exact source, model path,
threshold, and JSON target without recording or analyzing anything:

```bash
python3 Scripts/validate_microphone_vad.py \
  --dry-run \
  --duration 3 \
  --output .artifacts/microphone/mic-smoke.wav \
  --model-path Sources/SileroCoreML/Resources/SileroVAD.mlpackage
```

## Analyze an existing WAV

Existing-WAV mode is safest when microphone permissions or recorder tooling are
not available:

```bash
python3 Scripts/validate_microphone_vad.py \
  --input-wav .artifacts/microphone/existing-16khz-pcm.wav \
  --model-path Sources/SileroCoreML/Resources/SileroVAD.mlpackage \
  --threshold 0.5 \
  --json-output .artifacts/microphone/existing-16khz-pcm.summary.json
```

The WAV loader uses the stdlib path shared with the TEN-VAD evaluator. The file
must be uncompressed 16 kHz PCM WAV. Mono and stereo files are accepted; stereo
is averaged to mono. Audio is split into non-overlapping 512-sample frames and
any incomplete tail is dropped.

## Optional live recording

Live recording is intentionally explicit:

```bash
python3 Scripts/validate_microphone_vad.py \
  --duration 3 \
  --output .artifacts/microphone/mic-smoke.wav \
  --model-path Sources/SileroCoreML/Resources/SileroVAD.mlpackage \
  --threshold 0.5 \
  --json-output .artifacts/microphone/mic-smoke.summary.json
```

Before recording, the script prints a clear consent prompt showing the duration,
output path, and instruction to speak during the window. It uses a local recorder
command only if one is discoverable (`sox`, `rec`, `ffmpeg`, or `afrecord`). The
repository does not add AVFoundation/PyObjC capture APIs or new runtime audio
dependencies.

If no supported recorder command is found, the command returns exit code `2` and
asks you to use `--input-wav` or install/grant access to a supported local
recorder. If recording starts but fails on macOS, check:

```text
System Settings -> Privacy & Security -> Microphone
```

Grant access to Terminal or iTerm, rerun the command, or use `--input-wav`.

## Output metrics

The script runs the CoreML model through the same explicit-state helper contract
used by `Scripts/validate_coreml.py`: zero initial state/context, host-managed
64-sample context, and state carried between 512-sample frames. Thresholding is
inclusive: `probability >= threshold`.

The human-readable summary and optional JSON include:

- frame count and analyzed audio seconds;
- speech frame count, speech seconds, and speech ratio;
- consecutive speech segments with start, end, duration, and frames;
- probability min, mean, and max;
- inference seconds, RTF, RTFx, and latency mean/p50/p95/max.

Install optional validation dependencies from `Scripts/requirements.txt` only
when you want real CoreML inference. Parser, dry-run, and unit tests remain
dependency-light.
