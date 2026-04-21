# RMFT Source vs Round-2 AR Validation

## Background

Task G changes the repository's main validation axis from an old
round-1-vs-round-2 projection comparison to the AR-facing question:

`RMFT source-reference pairing tensors -> AR spectrum`

versus

`round-2 projected physical channels -> AR spectrum`

This is the validation axis that matters for the current forward-physics
repository because AR spectra are the downstream observable. Round-1 comparison
numbers remain useful historical context, but they are no longer the main
decision target.

## Source-Reference Path

The source-reference path is implemented in
`src/source/rmft_source_ar_validation.py`.

For each representative Luo sample, it:

- loads the raw Luo source pairing tensors,
- converts pairing tensors once from eV to meV through the existing source
  helper,
- applies the same round-2 gauge-fix convention,
- builds `Delta(k) = delta_x cos(kx) + delta_y cos(ky) + delta_z` directly from
  the gauge-fixed source tensors,
- runs the same `SpectroscopyPipeline` and multichannel BTK transport kernel as
  the round-2 projected model.

The comparison path is therefore not a new physics core. It is a diagnostic
adapter that lets the existing AR pipeline consume the source tensor pairing
structure as directly as the current code permits.

## Representative Audit

The generated output directory is:

`outputs/core/rmft_source_vs_round2_ar_validation/`

It contains:

- `rmft_source_vs_round2_ar_validation_summary.json`
- `rmft_source_vs_round2_ar_validation_metrics.csv`
- `rmft_source_vs_round2_best_scan.png`
- `rmft_source_vs_round2_median_scan.png`
- `rmft_source_vs_round2_worst_scan.png`

Representative samples are chosen by round-2 retained-ratio rank:

- best retained-ratio sample
- median retained-ratio sample
- worst retained-ratio sample

Each representative sample is scanned over:

- interface angle: `0`, `pi/8`, `pi/4`
- barrier strength: `0.25`, `0.5`, `1.0`
- broadening gamma: `0.5`, `1.0`, `2.0`
- temperature: `0.5 K`, `3 K`, `8 K`

The current full run has 36 comparison cases.

## AR Metrics

The summary reports AR-relevant discrepancy metrics:

- global spectrum MSE
- max and mean absolute conductance difference
- zero-bias conductance shift
- positive and negative peak-position shifts
- positive and negative peak-height shifts
- low-bias, shoulder-region, and high-bias MSE

Current generated summary:

- median source-vs-round2 MSE: `6.398474742484661e-06`
- p95 source-vs-round2 MSE: `0.003814198688302509`
- median absolute zero-bias shift: `0.008347276774758772`
- p95 absolute zero-bias shift: `0.1076348982838671`
- median max absolute conductance difference: `0.011052593784077525`
- worst-case max absolute conductance difference: `0.235154199695649`

The largest current discrepancy occurs for the median retained-ratio
representative under `barrier_z = 0.25`.

## Judgment

Under the current Task-G thresholds, the representative scan says the round-2
truth layer is sufficient for the present AR-facing forward work:

- median MSE is far below `1e-2`,
- median absolute zero-bias shift is below `0.1`,
- worst-case max absolute conductance difference stays below `0.25`.

This is an AR-facing adequacy statement, not a claim of full RMFT tensor
equivalence. Residual-anatomy diagnostics still show that the round-2 projection
does not retain the full source tensor. The present conclusion is narrower:

> The current round-2 truth layer appears to preserve enough of the RMFT source
> pairing structure for representative AR spectra in this audit, while some
> local low-barrier discrepancies remain visible and should be tracked before
> inverse-work claims are made.

## Status of Round-1 Comparisons

Round-1-vs-round-2 projection comparisons are now historical / compatibility
background only. They are not the repository's main validation target. Current
AR-facing decisions should cite the source-reference AR validation outputs above
and the round-2 residual-anatomy diagnostics, not old round-1 improvement
numbers.
