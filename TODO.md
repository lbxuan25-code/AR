# TODO

## Current Task

### Task H — Define the fit-layer parameterization for AR inversion
#### Goal
Define the parameter layer that should actually be inferred from experiment.

#### Rationale
The project goal is to infer the **most likely order-parameter features**, not to claim a unique microscopic RMFT parameter point.

#### Implement
- Separate clearly:
  - truth layer = full current round-2 physical channels
  - fit layer = lower-dimensional inversion control space
- Decide which quantities are free in the fit layer:
  - core round-2 pairing channels
  - weak optional channel policy
  - transport parameters
- Decide which quantities should be:
  - fixed
  - strongly regularized
  - reported only as uncertainty bands / candidate families
- Write the inversion output contract:
  - candidate clusters
  - parameter families
  - confidence-ranked solutions
  - never a single “true” point claim

#### Deliverables
- fit-layer design note
- parameter table with free / fixed / weak / derived fields
- updated project docs reflecting the truth-layer vs fit-layer split

#### Acceptance
Task H is complete only if a new developer can answer:
- what is the truth layer,
- what is the fit layer,
- what exactly will be inferred from AR spectra.

---

## Backlog

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
- ready to define the AR inversion fit layer in Task H
