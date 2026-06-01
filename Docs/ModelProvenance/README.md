# Model provenance

This directory records the chosen upstream source of truth for Silero VAD model
artifacts used during evaluation work for this repository.

## Current pinned upstream

- Upstream repository: `snakers4/silero-vad`
- Upstream release tag: `v6.2.1`
- Upstream commit: `7e30209a3e901f9842f81b225f3e93d8199902b1`
- Upstream package version: `6.2.1`
- Upstream license: MIT
- Provenance manifest: `silero-vad-v6.2.1.json`

## Repository policy

This repository now bundles only the validated converted CoreML runtime model.
Raw upstream artifacts and local conversion outputs remain ignored.

- Do not add raw upstream `.onnx`, `.jit`, or `.safetensors` files to git.
- Do not commit downloaded upstream artifacts into local working directories such
  as `.artifacts/` or `.generated/`.
- Use the manifest in this directory to identify the exact upstream files that
  should be fetched locally for validation or conversion work.
- Pin release tags and artifact SHA-256 values rather than moving `master`.

## Why this exists

The package needs a stable upstream reference for validation, conversion, and
notice review. Recording provenance here makes the selected source explicit and
keeps the bundled converted runtime model traceable to the official artifact.

## If bundling is ever proposed later

Before any additional Silero VAD model file is added to this repository:

1. Confirm the exact upstream artifact and digest details.
2. Recheck licensing and attribution requirements.
3. Update `THIRD_PARTY_NOTICES.md` with the required notice text.
4. Remove or narrow ignore rules only for the approved runtime resource path.

Raw upstream model artifacts should remain outside the repository even though the
validated converted runtime resource is bundled.
