# Current Output Index After Task K

## Purpose

Task K reduces the output tree to decision-relevant artifacts for the current
forward-physics repository. The goal is that a new reader can quickly identify
which generated files matter now and why.

The current validation story is:

`RMFT source tensors -> round-2 physical channels -> AR response`

not historical round-1 comparisons or training-process logs.

## Keep List

### Authoritative Baseline Provenance

`outputs/source/round2_baseline_selection.json`

- role: single authoritative formal round-2 baseline record;
- consumed by `src/core/formal_baseline.py`;
- read by `core.presets.base_physical_pairing_channels()`;
- must remain in sync with formal baseline tests.

### Source-To-Round2 Fidelity

`outputs/source/round2_projection_summary.json`

- role: current source-to-round2 projection summary;
- includes baseline cluster summary and current projection statistics;
- used by formal-baseline consistency tests.

`outputs/source/round2_projection_examples.csv`

- role: compact per-sample projection examples;
- useful for quick inspection of projected channels and retained ratios.

`outputs/source/round2_residual_anatomy_summary.json`

- role: residual-anatomy diagnosis for where round-2 still misses the source
  tensors;
- supports the statement that round-2 is an interpretable restricted
  projection, not full-RMFT-equivalent.

`outputs/source/round2_residual_examples.csv`

- role: representative residual examples.

`outputs/source/round2_residual_anatomy_heatmaps.png`

- role: aggregate residual heatmap visualization.

`outputs/source/round2_residual_representatives.png`

- role: representative residual visualization.

### AR-Facing Source-Reference Validation

`outputs/core/rmft_source_vs_round2_ar_validation/`

Files:

- `rmft_source_vs_round2_ar_validation_summary.json`
- `rmft_source_vs_round2_ar_validation_metrics.csv`
- `rmft_source_vs_round2_best_scan.png`
- `rmft_source_vs_round2_median_scan.png`
- `rmft_source_vs_round2_worst_scan.png`

Role:

- current main AR-facing validation axis;
- directly compares RMFT source-reference spectra against round-2 projected
  spectra;
- answers whether round-2 is sufficient for representative AR forward work.

### Formal Round-2 Baseline Spectral Validation

`outputs/core/round2_baseline_spectral_validation/`

Files:

- `round2_baseline_spectral_validation_summary.json`
- `round2_baseline_spectral_validation_metrics.csv`
- `round2_baseline_scan_comparison.png`
- `round2_channel_sensitivity_scan.png`

Role:

- validates what the formal round-2 baseline changes in AR spectra;
- includes channel-sensitivity checks for currently relevant round-2 levers;
- remains useful as a baseline-specific spectral audit.

### Stable Forward Interface Examples

`outputs/core/forward_interface/`

Files:

- `fit_layer_example_spectrum.json`
- `source_round2_example_spectrum.json`
- `fit_layer_inplane_110_example_spectrum.json`
- `fit_layer_inplane_110_spread_example_spectrum.json`
- `directional_capability_index.json`

Role:

- tiny examples for the Task-I stable forward interface;
- demonstrate canonical schema and metadata for external training repositories;
- publish the Task-Q directional capability contract in checked-in example
  payload form;
- should stay small and reproducible.

### Direction Capability Audit

`outputs/core/direction_capability_audit/`

Files:

- `direction_capability_summary.json`
- `direction_capability_metrics.csv`
- `direction_capability_representative_plots.png`

Role:

- Task-L audit of the current directional semantics;
- documents that `interface_angle` is a 2D in-plane interface-normal angle;
- separates supported in-plane high-symmetry shorthand from unsupported
  `c-axis` language.

### Directional Modes Validation

`outputs/core/directional_modes_validation/`

Files:

- `directional_modes_summary.json`
- `directional_modes_metrics.csv`
- `directional_modes_comparison.png`

Role:

- Task-M validation of callable high-symmetry in-plane directional modes;
- verifies `inplane_100` and `inplane_110` reproduce their raw-angle forward
  calls;
- confirms outputs record both `direction_mode` and `interface_angle`.

### Generic In-Plane Direction Scan

`outputs/core/inplane_direction_scan/`

Files:

- `summary.json`
- `metrics.csv`
- `inplane_direction_scan_plots.png`

Role:

- Task-N dense scan over `[0, pi/2]` raw in-plane angles;
- classifies generic non-high-symmetry angles as robust, caution, or unstable
  using quantitative thresholds;
- documents that generic raw in-plane angles are diagnostic / caution-required
  rather than broadly promoted truth modes.

### C-Axis Direction Audit

`outputs/core/c_axis_direction_audit/`

Files:

- `summary.json`
- `capability_matrix.csv`

Role:

- Task-O audit of whether true c-axis transport is physically implemented;
- records the blocking gaps: no `kz`, no out-of-plane velocity, no 3D/c-axis
  Fermi-surface construction, and no c-axis reflected-state path;
- documents that c-axis is formally unsupported and cannot be emulated by a
  raw 2D in-plane `interface_angle`.

### Directional Spread Validation

`outputs/core/directional_spread_validation/`

Files:

- `directional_spread_summary.json`
- `directional_spread_metrics.csv`
- `directional_spread_validation.png`

Role:

- Task-P validation of narrow spread-averaged spectra around supported in-plane
  named modes;
- verifies smooth spectral evolution as half width increases over
  `0`, `pi/128`, `pi/64`, and `pi/32`;
- documents the current uniform symmetric averaging rule and safe-use boundary.

### Required Local Luo Source Cache

`outputs/source/cache/RMFT_Ni327/`

Role:

- local copy of Luo RMFT source artifacts;
- required by `source.luo_loader` and source-dependent tests in the current
  network-restricted workspace;
- not a historical diagnostic output.

## Removed In Task K

The following generated outputs were removed because they are no longer current
decision targets:

- `outputs/source/ar_aware_projection_comparison_summary.json`
- `outputs/source/ar_aware_projection_examples.csv`
- `outputs/source/ar_aware_projection_representative_spectra.png`
- `outputs/core/round2_baseline_forward/round2_forward_summary.json`
- `outputs/core/round2_baseline_forward/round2_spectrum.png`

The old CLI entry points that regenerated those historical output locations
were also removed:

- `scripts/source/compare_ar_aware_projection.py`
- `scripts/core/run_round2_baseline_forward.py`

Rationale:

- AR-aware projection comparison was a Task-B exploratory diagnostic and is now
  superseded by the Task-G RMFT-source-reference vs round-2 AR validation axis;
- the baseline-forward demo was a lightweight runnable example superseded by
  the formal baseline spectral validation and the Task-I forward-interface
  examples.

The code paths and smoke tests for those diagnostic capabilities remain where
they are still useful. Task K only removes stale generated outputs from the
current output narrative.

## Current Output Tree

```text
outputs/
  core/
    c_axis_direction_audit/
    direction_capability_audit/
    directional_modes_validation/
    directional_spread_validation/
    forward_interface/
    inplane_direction_scan/
    rmft_source_vs_round2_ar_validation/
    round2_baseline_spectral_validation/
  source/
    cache/RMFT_Ni327/
    round2_baseline_selection.json
    round2_projection_examples.csv
    round2_projection_summary.json
    round2_residual_anatomy_heatmaps.png
    round2_residual_anatomy_summary.json
    round2_residual_examples.csv
    round2_residual_representatives.png
```

This is the compact set that matters for current scientific and interface
decisions.
