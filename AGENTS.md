# AGENTS.md

## Purpose

This repository is developed with Codex as a task executor, not as an autonomous planner.
Codex must follow the repository task documents exactly and work in **small verified steps**.

The top-level rule is:

> At any moment, only the single task in `TODO.md -> Current Task` may be executed.

`Backlog` tasks are **not active** until they are explicitly promoted into `Current Task`.

---

## Source of truth

When deciding what to do, use the following priority order:

1. `TODO.md`
2. current repository code and tests
3. current repository docs
4. historical outputs only if they are explicitly referenced by current docs/tests

If these sources conflict:

- stop
- report the conflict
- do **not** reinterpret the project on your own

---

## Task execution protocol

### 1. Read before doing anything
At the start of each work session:

- read `TODO.md`
- identify the **single active task** under `Current Task`
- ignore `Backlog` except for understanding future order
- do not plan beyond the current task unless needed for local implementation decisions

### 2. One active task only
Rules:

- execute only the single task in `Current Task`
- do **not** start any `Backlog` task early
- do **not** merge multiple tasks into one large change
- do **not** do extra cleanup, refactors, or documentation unless the active task explicitly requires it

### 3. Work in minimal valid increments
Prefer small coherent changes that can be verified immediately.

Good increments:
- one diagnostic module
- one projection improvement
- one validation script
- one documentation sync
- one cleanup pass

Bad increments:
- changing physics, docs, cleanup, outputs, and task promotion all at once without intermediate verification

### 4. Verification is mandatory
Before claiming a task is complete:

- run or update the relevant tests
- check that required outputs were generated
- confirm the active task's acceptance checklist is satisfied
- confirm no unrelated module was modified without necessity

If any acceptance item is not satisfied, the task is **not complete**.

### 5. Promotion rule
A task may move from `Backlog` to `Current Task` **only if**:

- all checklist items in the current task are complete
- outputs/tests/docs required by the current task are in place
- you have verified the current task is complete

Only then may you update `TODO.md` by:

1. moving the finished task out of `Current Task`
2. placing the next backlog task into `Current Task`
3. preserving the order of remaining backlog tasks

Do **not** promote multiple tasks at once.

---

## Required behavior when editing TODO.md

When a task is completed and verified:

- update `TODO.md`
- preserve the document structure:
  - `Current Task`
  - `Backlog`
  - `Archive`
- keep exactly **one** task under `Current Task`
- do not reorder backlog unless the task document explicitly requires it
- do not delete historical stage summaries from `Archive`

If the current task is incomplete, do **not** edit task ordering in `TODO.md`.

---

## Code modification boundaries

Unless the active task explicitly requires broader changes:

- do not modify normal-state logic while working on pairing-state tasks
- do not modify surrogate/inverse code during source-projection tasks
- do not rewrite BTK solver logic during pairing truth-layer tasks
- do not change repository-wide conventions unless required by the current task

Compatibility code may remain **only if** it is still required by:
- current tests
- current workflow
- current docs

Otherwise, obsolete code should eventually be removed under the cleanup task.

---

## Cleanup / decontamination policy

Old files can contaminate future work. Codex must avoid leaving stale paths behind.

### Never leave behind:
- `_old`, `_tmp`, `_bak`, duplicated draft files
- abandoned experimental scripts
- scratch notebooks
- superseded output artifacts
- outdated docs that describe old behavior as if it were current
- dead code paths that are no longer intended to be used

### Keep only if still needed:
- compatibility layers used by tests/workflow
- outputs referenced by current docs/tests
- current-stage diagnostics required by the active task

If the active task is not the cleanup task, do not perform broad cleanup unless necessary to keep the current task correct and unambiguous.

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

## Pairing-state specific guidance for the current project

This repository currently treats the **round-2 pairing truth layer** as the central physics truth representation.

Project-specific rules:

- do not revert from `PhysicalPairingChannels` back to a round-1-only worldview
- do not add many new pairing channels unless the active task's residual analysis justifies them
- treat `delta_zx_s` as an optional weak channel unless current-task evidence requires otherwise
- prefer physically interpretable reconstruction over opaque numerical compression
- keep the path
  `source pairing tensors -> physical channels -> Delta(k) -> BTK`
  clean and explicit

---

## If something is unclear

If you are unsure:

- inspect the active task in `TODO.md`
- inspect the current implementation and tests
- make the smallest change consistent with the current task
- if ambiguity remains, stop and report it instead of guessing broadly

---

## Summary rule

The repository should evolve like this:

- one active task at a time
- small verified changes
- no premature backlog execution
- no stale artifacts polluting future work
- update `TODO.md` only after verified task completion
