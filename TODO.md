# TODO

## Current Task

### Task I — Prepare the forward repository to serve an external training repository
#### Goal
Turn the current repository into a clean forward-physics source for later surrogate / inverse work in a separate repository.

#### Implement
- Define a stable forward interface:
  - canonical input schema
  - canonical output schema
  - canonical metadata fields
- Freeze the authoritative baseline / projection / channel conventions used for dataset generation
- Add minimal forward-facing scripts or callable entry points for:
  - generating spectra from fit-layer parameters
  - generating spectra from source-linked round-2 parameters
- Add version tags / metadata fields so external training runs can record which forward definition they used

#### Deliverables
- forward API / contract note
- minimal callable generation interface
- versioned metadata convention for datasets

#### Acceptance
Task I is complete only if a separate training repository could call this repository as a stable forward engine without copying its internals.

---

## Backlog

### Task J — Create the new surrogate / inverse repository plan
#### Goal
Design the new repository that will host dataset generation orchestration, surrogate training, and inverse training.

#### Implement
- Define the split of responsibilities between repositories:
  - current repository = forward truth chain
  - new repository = training / inversion / experiment fitting
- Specify the new repository structure:
  - dataset orchestration
  - training
  - evaluation
  - inverse search
  - experiment fitting outputs
- Define dependency strategy:
  - use this repository as an external forward dependency
  - do not duplicate the forward physics code
- Write the initial task list for the new repository

#### Deliverables
- new-repository design note
- initial `AGENTS.md`
- initial `TODO.md`
- initial directory plan

#### Acceptance
Task J is complete only if the new training repository could be created without ambiguity.

---

### Task K — Reduce historical outputs and keep only decision-relevant diagnostics
#### Goal
Further shrink the current repository outputs so they serve current scientific decisions rather than historical process logging.

#### Implement
- demote or remove non-essential historical comparison outputs
- keep only outputs that are still directly useful for:
  - source-to-round2 fidelity
  - round-2 spectral validation
  - authoritative baseline provenance
  - forward API correctness
- produce a compact index of “current important outputs”

#### Deliverables
- compact output index note
- reduced outputs tree
- cleanup summary

#### Acceptance
Task K is complete only if a new reader can quickly identify the few outputs that matter for current scientific decisions.

---

## Archive

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
- ready to prepare the stable forward interface in Task I
