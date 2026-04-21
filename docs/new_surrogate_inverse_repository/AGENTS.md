# AGENTS.md

## Purpose

This future repository will host AR surrogate training, inverse search, and
experiment-fitting orchestration.

It is not the source of truth for forward physics. The forward truth chain lives
in the LNO327 AR Forward Truth Chain repository and must be consumed through
its stable `forward` interface.

## Source Of Truth

Priority order:

1. this repository's `TODO.md`
2. dataset metadata and forward-interface versions
3. the external forward repository's public `forward` API contract
4. local training / inverse code and tests
5. local experiment-fitting reports

If local training code conflicts with forward-interface metadata, the forward
metadata wins.

## Repository Boundary

This repository owns:

- dataset orchestration;
- sampling policies for fit-layer and transport controls;
- surrogate model training;
- surrogate evaluation and calibration;
- inverse search;
- experiment-fitting reports;
- checkpoint and run metadata management.

This repository must not copy or fork:

- normal-state Hamiltonian code;
- round-2 pairing matrix code;
- RMFT source projection code;
- interface matching code;
- BTK transport solver code;
- authoritative formal baseline files.

Use the external forward repository as the forward engine.

## Required Forward Dependency

All generated data must record the forward interface metadata emitted by the
forward repository:

- `forward_interface_version`
- `output_schema_version`
- `pairing_convention_id`
- `formal_baseline_record`
- `formal_baseline_selection_rule`
- `projection_config`
- `git_commit`
- `git_dirty`

Datasets generated with different version metadata must be treated as distinct
dataset families.

## Execution Protocol

At any moment, only the single task in `TODO.md -> Current Task` may be
executed.

Do not start backlog tasks early. Do not combine dataset-generation,
surrogate-training, inverse-search, and experiment-fitting tasks into one
unverified mega-step.

Before marking a task complete:

- run relevant unit tests;
- verify generated artifacts exist;
- write or update run metadata;
- confirm forward-interface metadata is recorded;
- update `TODO.md` only after completion.

## Scientific Reporting Rules

Inverse outputs must be candidate families, not unique microscopic truth claims.

Reports should say:

- "the AR data are compatible with these feature families"

They should not say:

- "the order parameter is uniquely this RMFT point"

Always separate:

- fit-layer pairing controls;
- transport nuisance controls;
- experimental preprocessing;
- surrogate uncertainty;
- final forward recheck results.

## Storage Rules

Large artifacts belong in explicit output directories, not in source code paths:

- datasets under `outputs/datasets/`;
- checkpoints under `outputs/checkpoints/`;
- training logs under `outputs/runs/`;
- inverse results under `outputs/inverse/`;
- experiment fits under `outputs/experiments/`.

Do not commit heavyweight generated artifacts unless the project explicitly
decides they are small, canonical examples.
