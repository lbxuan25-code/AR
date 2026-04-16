# RMFT-Guided Surrogate for Pairing-and-Transport AR Spectra

## Project Goal

This repository is a greenfield surrogate project for Ni327 AR spectroscopy. It combines:

- a reusable physics forward workflow,
- a bridge from Luo's RMFT source data,
- a dataset builder that labels samples with the physics forward solver,
- a surrogate model for `pairing raw params + transport params -> AR spectrum`,
- and a surrogate-assisted inverse demo.

## Round-1 Scope

Round 1 only builds the minimum reproducible loop:

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

The baseline normal-state and pairing parameters are migrated from the local
`LNO327_AR_Phenomenology` repository and documented in
`docs/greenfield_repo_design.md`.
