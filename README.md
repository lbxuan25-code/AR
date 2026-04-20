# RMFT-Guided Surrogate for Pairing-and-Transport AR Spectra

## Project Goal

This repository is a greenfield surrogate project for Ni327 AR spectroscopy. It combines:

- a reusable physics forward workflow,
- a bridge from Luo's RMFT source data,
- a dataset builder that labels samples with the physics forward solver,
- a surrogate model for `pairing raw params + transport params -> AR spectrum`,
- and a surrogate-assisted inverse demo.

## Current Status

The repository now has a Stage-3 round-2 pairing truth layer on top of the
original round-1 forward loop. The active physics path is:

`Luo source pairing tensors -> PhysicalPairingChannels -> Delta(k) -> interface diagnostics -> multichannel BTK spectrum`

The round-2 truth layer uses seven core channels plus one optional weak channel
(`delta_zx_s`). The formal round-2 baseline is sourced from
`outputs/source/round2_baseline_selection.json`, which is generated from the
low-temperature charge-balanced Luo cluster and read by
`core.presets.base_physical_pairing_channels()`.

The original round-1 work still provides the minimum reproducible loop:

1. initialize the repository and baseline configuration;
2. migrate or reproduce the physics core;
3. inspect and bridge the Luo RMFT source;
4. build a pairing+transport dataset;
5. train and evaluate a lightweight surrogate;
6. run a surrogate-assisted inverse demo.

## Core Principle

The physics forward workflow is the ground-truth path:

`ModelParams -> FS / projection / interface diagnostics -> multichannel BTK spectrum`

The surrogate is only an approximator of this workflow and never replaces it.

## Inverse Output Contract

Round-1 inverse outputs must be candidate families rather than a unique truth:

- top-K candidate families,
- parameter clusters,
- near-optimal solution sets.

The project does not claim a single true parameter point from inverse fitting.

## Baseline Provenance

The baseline normal-state parameters remain the repository-local fixed normal
state in `src/core/presets.py`.

The formal round-2 pairing baseline is not handwritten in `presets.py`. It is
read from the authoritative source-side record
`outputs/source/round2_baseline_selection.json`; see
`docs/pairing_state_stage3.md` for the current pairing-state design.
