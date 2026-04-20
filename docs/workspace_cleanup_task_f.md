# Workspace Cleanup Task F

## Scope

Task F removed stale files that could make the current round-2 truth-layer path
ambiguous. The cleanup did not change normal-state logic, BTK solvers, source
projection formulas, or the surrogate / inverse library modules.

## Authoritative Current Path

The current round-2 truth-layer path is:

`Luo source tensors -> round2_projection.py -> PhysicalPairingChannels -> Delta(k) -> BTK`

The formal baseline has one authoritative source:

`outputs/source/round2_baseline_selection.json`

Runtime access goes through `src/core/formal_baseline.py` and
`core.presets.base_physical_pairing_channels()`.

## Deleted Files

Historical round-1 / greenfield docs removed because they no longer describe
the current implementation path:

- `docs/greenfield_repo_design.md`
- `docs/luo_source_map_round1.md`
- `docs/projection_consistency_round1.md`
- `docs/surrogate_round1_design.md`

Obsolete CLI entry points removed because they generated stale round-1 or
non-authoritative artifacts:

- `scripts/source/check_projection_consistency.py`
- `scripts/source/inspect_luo_source.py`
- `scripts/dataset/build_pairing_transport_dataset.py`
- `scripts/surrogate/train_pairing_transport_surrogate.py`
- `scripts/surrogate/evaluate_pairing_transport_surrogate.py`
- `scripts/inverse/run_inverse_with_surrogate.py`

Obsolete diagnostic module and test removed:

- `src/source/projection_diagnostics.py`
- `tests/source/test_projection_consistency_smoke.py`

Obsolete generated outputs removed:

- `outputs/source/luo_source_inspection_summary.json`
- `outputs/source/projection_consistency_summary.json`
- `outputs/source/projection_consistency_examples.csv`
- `outputs/source/projection_consistency_hist.png`
- `outputs/source/projection_residual_scatter.png`
- `outputs/dataset/`
- `outputs/surrogate/`
- `outputs/inverse/`

Python bytecode caches under `src/` and `tests/` were also removed.

## Rewritten Files

- `README.md`: now points to the current Stage-3 round-2 truth layer and no
  longer references deleted historical docs.
- `docs/pairing_state_stage3.md`: current authoritative design note.
- `docs/order_parameter_refactor_round2.md`: current round-2 refactor summary.
- `TODO.md`: updated after verified completion of Task F.

## Intentionally Preserved

Compatibility and diagnostic code kept because it is still used by current tests,
current source metrics, or current Stage-3 diagnostics:

- `PairingParams` compatibility in `src/core/parameters.py` and `src/core/pairing.py`
- `compatibility_physical_pairing_channels()` for explicit legacy comparison
- `src/source/luo_projection.py` because round-1 metrics are still used in
  current round-1-vs-round-2 comparison diagnostics
- `src/data/` and `src/surrogate/` library modules because their smoke tests
  still exercise the legacy accelerator layer; they are not the authoritative
  round-2 truth layer
- `outputs/source/cache/RMFT_Ni327` because the local source loader and tests
  depend on this cache in a network-restricted environment
- current round-2 outputs:
  `round2_baseline_selection`, `round2_projection`, `round1_vs_round2`,
  residual anatomy, AR-aware comparison, and Task-D spectral validation outputs

## Result

The workspace now has one authoritative current round-2 truth-layer path, with
historical round-1 artifacts removed from docs, scripts, and outputs.
