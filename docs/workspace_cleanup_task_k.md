# Workspace Cleanup Task K

## Scope

Task K reduced generated outputs so the repository points to current scientific
decisions rather than historical process artifacts.

No normal-state logic, round-2 projection logic, pairing core, interface
diagnostics, BTK solver, formal baseline definition, or forward-interface code
was changed.

## Kept Output Categories

The compact current output set is indexed in
`docs/current_output_index_task_k.md`.

Kept categories:

- authoritative baseline provenance;
- source-to-round2 fidelity and residual anatomy;
- RMFT-source-reference vs round-2 AR validation;
- formal round-2 baseline spectral validation;
- stable forward-interface examples;
- required local Luo source cache.

## Removed Outputs

Removed historical / non-current generated outputs:

- `outputs/source/ar_aware_projection_comparison_summary.json`
- `outputs/source/ar_aware_projection_examples.csv`
- `outputs/source/ar_aware_projection_representative_spectra.png`
- `outputs/core/round2_baseline_forward/round2_forward_summary.json`
- `outputs/core/round2_baseline_forward/round2_spectrum.png`
- empty directory `outputs/core/round2_baseline_forward/`

Removed historical CLI entry points that would regenerate those stale output
locations:

- `scripts/source/compare_ar_aware_projection.py`
- `scripts/core/run_round2_baseline_forward.py`

Follow-up forward-interface metadata cleanup also regenerated:

- `outputs/core/forward_interface/fit_layer_example_spectrum.json`
- `outputs/core/forward_interface/source_round2_example_spectrum.json`

Those examples now use repository-relative baseline metadata and
`git_dirty = false`.

## Rationale

The Task-B AR-aware projection comparison remains useful as code-level
diagnostic capability and is still covered by smoke tests, but its generated
outputs are no longer part of the current decision narrative.

The old round-2 baseline-forward demo was superseded by:

- `outputs/core/round2_baseline_spectral_validation/`
- `outputs/core/forward_interface/`

The main current AR-facing claim should be read from:

- `docs/rmft_source_vs_round2_ar_validation.md`
- `outputs/core/rmft_source_vs_round2_ar_validation/`

## Result

A new reader can now scan `docs/current_output_index_task_k.md` and identify the
few output locations that matter for current decisions without sorting through
historical process logs.
