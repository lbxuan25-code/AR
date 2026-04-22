# TODO

## Current Task

### No active task — awaiting next assignment

All tasks currently listed in this TODO have been completed and archived.

---

## Backlog

---

## Archive

### Task Q — Publish the complete directional capability contract for external repositories

Completed and verified.

- Added `docs/directional_capability_contract_task_q.md` as the single external
  direction capability contract for the current forward interface.
- Added `outputs/core/forward_interface/directional_capability_index.json` as a
  compact machine-readable reference for external callers.
- Updated `docs/forward_interface_task_i.md`, `docs/current_output_index_task_k.md`,
  and `README.md` to point to the Task-Q contract.
- Extended `scripts/core/generate_forward_spectrum.py` so the stable CLI can
  generate narrow directional-spread spectra through the public forward
  helpers.
- Refreshed raw forward-interface examples and added checked-in examples for
  `inplane_110` and `inplane_110` directional spread.
- Confirmed the stable external boundary: named `inplane_100` / `inplane_110`
  modes are supported, generic raw in-plane angles remain diagnostic /
  caution-only, c-axis is unsupported, and spread is supported only around
  named in-plane modes.
- Added `tests/forward/test_directional_capability_contract_examples.py` to
  lock the contract index and example payload provenance.
- Verified with
  `PYTHONPATH=src pytest tests/forward/test_directional_capability_contract_examples.py tests/forward/test_directional_spread_smoke.py tests/forward/test_directional_modes_smoke.py tests/forward/test_forward_interface_smoke.py -q`.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Task P — Add directional spread primitives to the forward truth chain

Completed and verified.

- Added `DirectionalSpread`, spread sampling helpers, and spread validation to
  `src/forward/directions.py`.
- Added `generate_spread_spectrum_from_fit_layer(...)` and
  `generate_spread_spectrum_from_source_round2(...)` to the stable `forward`
  engine.
- Exported the spread helpers through the public `forward` package.
- Defined the current spread primitive as a uniform symmetric average around
  supported named in-plane modes with maximum half width `pi/32`.
- Added `src/core/directional_spread_validation.py` and
  `scripts/core/validate_directional_spread.py`.
- Generated `outputs/core/directional_spread_validation/` with summary,
  metrics CSV, and validation plot.
- Validated spread evolution across `inplane_100`, `inplane_110`, half widths
  `0`, `pi/128`, `pi/64`, `pi/32`, barriers `0.5` / `1.0`, and two pairing
  states.
- Found smooth width evolution with max observed width-step spectrum difference
  `0.20857708180230394` below the `0.25` threshold.
- Added `docs/directional_spread_task_p.md` and updated README / forward
  interface / current output index docs.
- Added `tests/forward/test_directional_spread_smoke.py` and
  `tests/core/test_directional_spread_validation_smoke.py`.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Task O — Decide and implement the true c-axis directional path (or explicitly wall it off)

Completed and verified.

- Added `src/core/c_axis_direction_audit.py` and
  `scripts/core/audit_c_axis_direction.py`.
- Generated `outputs/core/c_axis_direction_audit/` with an unsupported summary
  and capability matrix.
- Added `docs/c_axis_direction_task_o.md`.
- Audited the current normal-state, pairing, BdG, simulation model,
  Fermi-surface, interface-normal, and velocity paths.
- Found blocking gaps for true c-axis transport: no `kz`, no out-of-plane
  velocity, no 3D/c-axis Fermi-surface construction, and no c-axis
  reflected-state path.
- Formally marked c-axis as unsupported in the current forward truth chain.
- Updated `forward.directions` so `c_axis` / `c-axis` aliases raise a specific
  unsupported error rather than being silently mapped to any raw in-plane angle.
- Updated `README.md`, `docs/forward_interface_task_i.md`,
  `docs/current_output_index_task_k.md`, and prior direction docs.
- Added `tests/core/test_c_axis_direction_audit_smoke.py`.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Task N — Validate non-high-symmetry in-plane directions and define their support boundary

Completed and verified.

- Added `src/core/inplane_direction_scan.py` and
  `scripts/core/validate_inplane_directions.py`.
- Generated `outputs/core/inplane_direction_scan/` with dense scan summary,
  metrics CSV, and angular smoothness / matching plots.
- Added `docs/inplane_generic_direction_validation_task_n.md`.
- Scanned 33 raw 2D in-plane angles over `[0, pi/2]`.
- Measured nominal/tight/loose reflected-state matching, same-band retention,
  reflected mismatch statistics, tolerance sensitivity, and nearest-neighbor
  spectral smoothness.
- Defined explicit robust / caution / unstable thresholds.
- Found that the 3 high-symmetry checkpoints are robust, while the 30 generic
  non-high-symmetry angles contain 0 robust, 20 caution, and 10 unstable cases.
- Concluded that generic raw in-plane angles remain diagnostic /
  caution-required rather than broadly promoted truth modes.
- Updated `README.md`, `docs/forward_interface_task_i.md`, and
  `docs/current_output_index_task_k.md`.
- Added `tests/core/test_inplane_direction_scan_smoke.py`.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Task M — Build canonical directional forward modes for in-plane high-symmetry transport

Completed and verified.

- Added `src/forward/directions.py` with canonical `inplane_100` and
  `inplane_110` directional modes.
- Exported directional helpers through the stable `forward` package:
  `list_directional_modes()`, `interface_angle_for_direction_mode()`,
  `transport_with_direction_mode()`, and `replace_direction_mode()`.
- Updated `TransportControls` to record optional `direction_mode` provenance
  alongside raw `interface_angle`.
- Added forward-engine validation so inconsistent `direction_mode` /
  `interface_angle` combinations raise `ValueError`.
- Updated `scripts/core/generate_forward_spectrum.py` with
  `--direction-mode`.
- Added `src/core/directional_modes_validation.py` and
  `scripts/core/validate_directional_modes.py`.
- Generated `outputs/core/directional_modes_validation/` with summary, metrics,
  and raw-vs-named comparison plots.
- Added `docs/directional_modes_task_m.md` and updated README / forward
  interface / current output index docs.
- Refreshed forward-interface example outputs so transport provenance includes
  `direction_mode`.
- Added `tests/forward/test_directional_modes_smoke.py` and updated existing
  forward-interface smoke assertions.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Task L — Audit and standardize directional semantics in the forward truth chain

Completed and verified.

- Added `src/core/direction_capability_audit.py` and
  `scripts/core/audit_direction_capability.py`.
- Generated `outputs/core/direction_capability_audit/` with JSON summary, CSV
  metrics, and representative spectra / matching plots.
- Added `docs/direction_capability_task_l.md`.
- Defined `interface_angle` as a strictly 2D in-plane interface-normal angle.
- Classified in-plane `100` and `110` as supported high-symmetry raw-angle
  shorthand, generic in-plane angles as computable but caution-required, and
  `c-axis` as unsupported in the current model.
- Updated `README.md`, `docs/forward_interface_task_i.md`,
  `docs/current_output_index_task_k.md`, and the public forward schema docstring.
- Added `tests/core/test_direction_capability_audit_smoke.py`.
- Verified with `PYTHONPATH=src pytest tests -q`.

### Completed and verified previously
- Stage_1 — Independent forward-physics repository
- Stage_2 — Luo / RMFT source bridge and round-1 audit
- Stage_3 — Round-2 order-parameter truth-layer refactor
- Task A — Residual anatomy audit
- Task B — AR-aware projection test
- Task C — Freeze weak optional channel by default
- Task D — Spectral validation of the formal round-2 baseline
- Task E — Documentation sync for Stage-3 implementation
- Task F — Workspace cleanup / decontamination
- Task G — Rebuild the validation axis around RMFT-source-to-AR fidelity
- Task H — Define the fit-layer parameterization for AR inversion
- Task I — Prepare the forward repository to serve an external training repository
- Task J — Create the new surrogate / inverse repository plan
- Task K — Reduce historical outputs and keep only decision-relevant diagnostics

#### Current repository status before the new directionality cycle
- one authoritative round-2 forward truth path
- one authoritative formal baseline source
- stable external `forward` interface
- RMFT-source-reference vs round-2 AR validation established
- fit-layer parameterization documented
- external training / inverse repository already separated
- no active task before beginning the directionality cycle

#### Directionality-cycle objective
Build a complete, explicit, externally consumable directional forward capability, starting from semantic audit and ending in a stable direction contract for downstream repositories.
