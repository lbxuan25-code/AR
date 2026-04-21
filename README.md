# RMFT-Guided Surrogate for Pairing-and-Transport AR Spectra

## Project Goal

This repository is the forward-physics truth-chain project for Ni327 AR spectroscopy. It provides:

- a reusable physics forward workflow,
- a bridge from Luo's RMFT source data,
- the RMFT-source-to-round-2 projection logic,
- an authoritative formal round-2 baseline,
- and AR-facing validation diagnostics.

Large-scale surrogate training, inverse training, checkpoint management, and
experiment-fitting orchestration are intentionally deferred to a later separate
training / inversion repository.

## Current Status

The repository now has a Stage-3 round-2 pairing truth layer. The active physics
path is:

`Luo source pairing tensors -> PhysicalPairingChannels -> Delta(k) -> interface diagnostics -> multichannel BTK spectrum`

The round-2 truth layer uses seven core channels plus one optional weak channel
(`delta_zx_s`). The formal round-2 baseline is sourced from
`outputs/source/round2_baseline_selection.json`, which is generated from the
low-temperature charge-balanced Luo cluster and read by
`core.presets.base_physical_pairing_channels()`.

The main validation axis is now:

`RMFT source-reference AR spectra -> round-2 projected-channel AR spectra`

See `docs/rmft_source_vs_round2_ar_validation.md` and
`outputs/core/rmft_source_vs_round2_ar_validation/` for the current AR-facing
adequacy check. Round-1 comparisons remain historical / compatibility context
only.

## Core Principle

The physics forward workflow is the ground-truth path:

`ModelParams -> FS / projection / interface diagnostics -> multichannel BTK spectrum`

Future surrogate or inverse layers must remain downstream approximations of
this workflow and must not replace the forward truth chain in this repository.

## Baseline Provenance

The baseline normal-state parameters remain the repository-local fixed normal
state in `src/core/presets.py`.

The formal round-2 pairing baseline is not handwritten in `presets.py`. It is
read from the authoritative source-side record
`outputs/source/round2_baseline_selection.json`; see
`docs/pairing_state_stage3.md` for the current pairing-state design.
