# TODO

## Current Task

### No active task — awaiting next assignment
All tasks currently listed in this file have been completed and verified.

---

## Backlog

## Archive

### Task K — Reduce historical outputs and keep only decision-relevant diagnostics

Completed and verified.

- Added `docs/current_output_index_task_k.md` as the compact index of current
  decision-relevant generated outputs.
- Added `docs/workspace_cleanup_task_k.md` as the cleanup summary.
- Removed stale generated AR-aware projection comparison outputs from
  `outputs/source/`.
- Removed the superseded baseline-forward demo output directory from
  `outputs/core/`.
- Kept current outputs for authoritative baseline provenance,
  source-to-round2 fidelity, residual anatomy, RMFT-source-reference AR
  validation, formal round-2 baseline spectral validation, forward-interface
  examples, and the required Luo source cache.
- Removed Python bytecode caches after testing.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Task J — Create the new surrogate / inverse repository plan

Completed and verified.

- Added `docs/new_surrogate_inverse_repository_plan.md`.
- Added future-repository starter templates under
  `docs/new_surrogate_inverse_repository/`:
  `AGENTS.md`, `TODO.md`, and `DIRECTORY_PLAN.md`.
- Defined the repository split:
  current repository = forward truth chain;
  future repository = dataset orchestration, surrogate training, inverse search,
  and experiment fitting.
- Defined the dependency strategy: the future repository consumes this
  repository through the stable `forward` package or
  `scripts/core/generate_forward_spectrum.py` and must not copy forward physics
  internals.
- Specified the initial new-repository task sequence and directory ownership
  plan.
- Preserved this repository's root `README.md`, `AGENTS.md`, and template-free
  root files; only this `TODO.md` was minimally updated for task promotion.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Task I — Prepare the forward repository to serve an external training repository

Completed and verified.

- Added stable public package `forward` with canonical schemas and callable
  spectrum-generation entry points.
- Added `FitLayerSpectrumRequest` and
  `generate_spectrum_from_fit_layer(...)` for Task-H fit-layer controls.
- Added `SourceRound2SpectrumRequest` and
  `generate_spectrum_from_source_round2(...)` for Luo samples projected through
  the default round-2 truth layer.
- Added CLI entry point `scripts/core/generate_forward_spectrum.py`.
- Added version tags and canonical metadata fields:
  `ar_forward_v1`, `ar_forward_input_v1`, `ar_forward_output_v1`, and
  `round2_physical_channels_task_h_fit_layer_v1`.
- Generated example payloads under `outputs/core/forward_interface/`.
- Added `docs/forward_interface_task_i.md` and updated README / Task-H docs.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Task H — Define the fit-layer parameterization for AR inversion

Completed and verified.

- Added `docs/fit_layer_parameterization_task_h.md`.
- Defined the truth layer as the full current round-2 `PhysicalPairingChannels`
  forward representation with the authoritative formal baseline.
- Defined the fit layer as a lower-dimensional, gauge-fixed, regularized AR
  inversion control layer around the formal baseline.
- Added a parameter table separating free, strongly regularized,
  fixed-by-default weak, fixed, nuisance, and derived fields.
- Specified the inversion output contract as ranked candidate families with
  uncertainty ranges, not a unique microscopic RMFT point.
- Updated `README.md`, `docs/pairing_state_stage3.md`,
  `docs/order_parameter_refactor_round2.md`, and
  `docs/rmft_source_vs_round2_ar_validation.md`.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Task G — Rebuild the validation axis around RMFT-source-to-AR fidelity

Completed and verified.

- Added a direct RMFT source-reference AR comparison path against round-2 projected-channel AR spectra.
- Generated representative best / median / worst sample comparisons over interface angle, barrier strength, broadening, and temperature.
- Wrote current outputs under `outputs/core/rmft_source_vs_round2_ar_validation/`.
- Added `docs/rmft_source_vs_round2_ar_validation.md` and updated current docs so the main validation story is no longer round-1-centered.
- Removed the standalone round-1-vs-round-2 comparison script and generated JSON artifact from the current validation target set while retaining minimal compatibility checks in code/tests.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Completed previously
- Stage_1 — Independent forward-physics repository
- Stage_2 — Luo / RMFT source bridge and round-1 audit
- Stage_3 — Round-2 order-parameter truth-layer refactor
- Task A — Residual anatomy audit
- Task B — AR-aware projection test
- Task C — Freeze weak optional channel by default
- Task D — Spectral validation of the formal round-2 baseline
- Task E — Documentation sync for Stage-3 implementation
- Task F — Workspace cleanup / decontamination

#### Current repository status after those completed tasks
- one authoritative round-2 forward truth path
- one authoritative formal baseline source
- cleaned workspace
- RMFT-source-reference vs round-2 AR validation is now the main validation axis
- AR inversion fit-layer parameterization is documented
- stable forward interface is documented and callable
- separate surrogate / inverse repository plan is documented
- compact current output index is documented
- no pending tasks remain in this TODO
