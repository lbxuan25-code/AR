
# AGENTS.md

## Purpose

This repository is a **forward-physics truth-chain repository** for the LNO327 / La3Ni2O7 AR project.

Codex must treat this repository as the authoritative place for:
- the forward normal-state + pairing + interface + BTK physics chain,
- the RMFT-source-to-round-2 projection logic,
- the formal round-2 baseline,
- and the current AR-facing validation diagnostics.

This repository is **not** the main place for large-scale surrogate training, inverse training, checkpoint management, or experiment-fitting orchestration. Those belong in a later separate training / inversion repository.

The top-level execution rule remains:

> At any moment, only the single task in `TODO.md -> Current Task` may be executed.

`Backlog` tasks are not active until they are explicitly promoted into `Current Task`.

---

## Source of truth

When deciding what to do, use the following priority order:

1. `TODO.md`
2. current repository code and tests
3. current repository docs
4. current authoritative outputs referenced by current docs/tests

If these sources conflict:

- stop
- report the conflict clearly
- do not reinterpret the project on your own

---

## Repository role

This repository currently exists to answer questions of the following kind:

- What is the current forward-physics truth path?
- How is RMFT source information projected into the round-2 pairing truth layer?
- What does the current round-2 truth layer preserve or lose?
- Is the current round-2 truth layer sufficient for AR-facing work?
- What stable forward interface should an external training repository depend on?

This repository should **not** drift back into becoming a general experimentation dump for:
- old round-1 comparisons as the main narrative
- large training logs
- surrogate checkpoints
- inverse search outputs
- duplicated forward code for training workflows

---

## Current scientific direction

The main validation axis is now:

> **RMFT-source-reference -> AR response**  
> versus  
> **round-2 projected channels -> AR response**

The main question is no longer:

> “Is round-2 better than round-1?”

The main question is now:

> **“For the Andreev-reflection process, does the current round-2 truth layer preserve enough information from the original RMFT source?”**

Round-1 content may remain only as:
- historical context,
- compatibility support,
- minimal smoke-test coverage,
- or lightweight comparison background.

Round-1 outputs and narratives must not be treated as the main scientific validation target anymore.

---

## Task execution protocol

### 1. Read before doing anything
At the start of each work session:

- read `TODO.md`
- identify the single active task under `Current Task`
- ignore `Backlog` except for understanding later order
- do not plan beyond the current task unless needed for local implementation decisions

### 2. One active task only
Rules:

- execute only the single task in `Current Task`
- do not start any `Backlog` task early
- do not merge multiple tasks into one large change
- do not do unrelated refactors, cleanup, training, or documentation work unless the active task explicitly requires it

### 3. Work in minimal valid increments
Prefer small coherent increments that can be verified immediately.

Good increments:
- one validation path
- one diagnostic module
- one metrics output
- one focused doc update
- one forward-interface contract note
- one cleanup pass tightly tied to the active task

Bad increments:
- changing validation logic, docs, repository role, outputs, and task promotion all at once without intermediate verification

### 4. Verification is mandatory
Before claiming a task is complete:

- run or update the relevant tests
- check that required outputs were generated
- confirm the active task's acceptance checklist is satisfied
- confirm no unrelated module was modified without necessity

If any acceptance item is not satisfied, the task is not complete.

### 5. Promotion rule
A task may move from `Backlog` to `Current Task` only if:

- all checklist items in the current task are complete
- required outputs/tests/docs are in place
- you have verified the current task is complete

Only then may you update `TODO.md` by:

1. moving the finished task out of `Current Task`
2. placing the next backlog task into `Current Task`
3. preserving the order of remaining backlog tasks

Do not promote multiple tasks at once.

---

## Required behavior when editing TODO.md

When a task is completed and verified:

- update `TODO.md`
- preserve the document structure:
  - `Current Task`
  - `Backlog`
  - `Archive`
- keep exactly one task under `Current Task`
- do not reorder backlog unless the task document explicitly requires it
- do not delete completed task summaries from `Archive`

If the current task is incomplete, do not edit task ordering in `TODO.md`.

---

## Physics and modeling boundaries

Unless the active task explicitly requires broader changes:

- do not redesign the round-2 channel language casually
- do not expand the normal-state model casually
- do not rewrite the BTK solver casually
- do not change the formal baseline definition casually
- do not mix the forward truth layer with training-specific approximations

Important modeling rule:

> The truth layer and the fit layer are not the same thing.

The round-2 physical channels are currently the truth-layer representation.
A future fit layer may be lower-dimensional for inversion purposes.
Do not collapse these two concepts unless the active task explicitly defines that change.

---

## RMFT-source and round-2 guidance

The repository currently treats the following path as authoritative:

`RMFT source tensors -> round-2 physical channels -> Delta(k) -> interface diagnostics -> BTK spectrum`

Important project-specific rules:

- do not revert to a round-1-centered worldview
- do not present round-1-vs-round-2 comparison as the main scientific validation story
- do not add new pairing channels unless current-task evidence clearly justifies them
- treat `delta_zx_s` as an optional weak channel unless current-task evidence requires otherwise
- keep the formal baseline single-sourced and authoritative
- prefer physically interpretable structure over opaque numerical compression

---

## Validation guidance

When building diagnostics, prefer outputs that directly answer the current scientific question.

High-value diagnostics are those that clarify:
- how close round-2 AR spectra are to source-reference AR spectra,
- which source structures still matter for AR,
- which pairing channels truly move AR observables,
- whether current limitations are harmless or blocking for future inverse work.

Lower-value diagnostics are those that only restate old round-1-vs-round-2 comparisons without informing present scientific decisions.

---

## Forward-repository boundary vs future training repository

This repository should be prepared to serve as a stable dependency for a later separate training / inversion repository.

Therefore:

- do not place large-scale surrogate training loops here
- do not place inverse training orchestration here
- do not store large checkpoint trees here
- do not duplicate forward physics code for external training use

Instead, work here should aim to provide:
- a stable forward interface,
- stable baseline / projection / channel conventions,
- stable metadata for external dataset generation,
- and authoritative forward diagnostics.

---

## Cleanup / decontamination policy

Old files can contaminate future work. Codex must avoid leaving stale paths behind.

### Never leave behind:
- `_old`, `_tmp`, `_bak`, duplicated draft files
- abandoned experimental scripts
- scratch notebooks
- superseded outputs that no longer support current decisions
- outdated docs that present old validation logic as current
- dead code paths no longer intended for use

### Keep only if still needed:
- compatibility layers used by current tests/workflow
- outputs referenced by current docs/tests
- authoritative baseline records
- current-stage diagnostics that support the active task
- minimal historical context needed to explain repository evolution

If the active task is not a cleanup task, do not perform broad cleanup unless necessary to keep the current task correct and unambiguous.

---

## Reporting format for each task step

After each meaningful step, report clearly:

1. which files were changed
2. which outputs were created or updated
3. what was verified
4. whether the current task acceptance checklist is now fully satisfied
5. whether `TODO.md` was updated

If the task is not yet complete, explicitly say which checklist items remain open.

---

## If something is unclear

If you are unsure:

- inspect the active task in `TODO.md`
- inspect the current implementation and tests
- make the smallest change consistent with the active task
- if ambiguity remains, stop and report it instead of guessing broadly

---

## Summary rule

The repository should evolve like this:

- one active task at a time
- small verified changes
- no premature backlog execution
- no drift back to round-1-centered validation
- no mixing of truth-layer code with large training workflows
- no stale artifacts polluting future work
- update `TODO.md` only after verified task completion
