# Pairing State Stage 3

## Goal

Stage 3 upgrades the round-2 pairing layer from a merely runnable framework to
a more formal truth-layer candidate for BTK forward calculations.

The emphasis is not on adding more channels. It is on:

1. fixing the formal channel hierarchy,
2. making the source projection gauge-consistent and regularized,
3. defining a real round-2 baseline,
4. unifying reconstruction metrics across diagnostics.

## Finalized Channel Hierarchy

The full round-2 channel layer still contains 8 complex channels:

- `delta_zz_s`
- `delta_zz_d`
- `delta_xx_s`
- `delta_xx_d`
- `delta_zx_s`
- `delta_zx_d`
- `delta_perp_z`
- `delta_perp_x`

Stage 3 now distinguishes:

- core truth-layer channels:
  `delta_zz_s`, `delta_zz_d`, `delta_xx_s`, `delta_xx_d`,
  `delta_zx_d`, `delta_perp_z`, `delta_perp_x`
- optional weak channel:
  `delta_zx_s`

The optional channel is still supported in code, but it is no longer treated as
co-equal with the seven core channels.

## Constrained Projection

`src/source/round2_projection.py` no longer uses plain unconstrained
`np.linalg.lstsq(A, b)`.

Stage 3 projection now performs:

1. source tensors converted once from eV to meV,
2. a global gauge fix before fitting,
3. weighted reconstruction with block weights
   `w_x = 1.0`, `w_y = 1.0`, `w_z = 1.15`,
4. channel-dependent ridge regularization.

Regularization strengths are intentionally weakest on the main channels and
strongest on the optional weak mixed channel:

- `delta_zx_s`: `5.0e-3`, followed by a default soft-freeze decision gate
- `delta_zx_d`, `delta_perp_x`: `7.5e-4`
- `delta_zz_s`, `delta_xx_s`, `delta_xx_d`, `delta_perp_z`: `1.0e-4`
- `delta_zz_d`: `2.0e-4`

The global gauge reference is chosen from the prioritized core-channel list.
On the current Luo source cache, the anchor is always `delta_zz_s`.

Task C default policy further treats `delta_zx_s` as a weak optional channel:

- the code still supports it,
- the full fit is still computed,
- but the default workflow keeps it frozen at zero unless it clears a
  "clear need" gate:
  - relative magnitude at least `8e-2` compared with the strongest core channel
  - residual-norm reduction at least `1e-2` when compared against the
    frozen-channel refit

So the default truth model is now explicitly centered on the 7 core channels,
while `delta_zx_s` remains available for non-default diagnostic runs.

## Unified Metrics

Stage 3 standardizes the repository-wide projection metrics in
`src/source/projection_metrics.py`.

The unique definitions are:

- `residual_norm_total = ||source - recon||_F`
- `retained_ratio_total = 1 - residual_norm_total / source_norm_total`
- `omitted_fraction_total = residual_norm_total / source_norm_total`

The same definitions are now used by:

- round-1 projection consistency diagnostics
- round-2 projection diagnostics
- round-1 vs round-2 comparison

This removes the old ambiguity where retained ratio could mean either
`recon/source` or `1 - residual/source`.

## Formal Round-2 Baseline

`core.presets.base_physical_pairing_channels()` is now a real round-2 baseline,
not a translated round-1 compatibility baseline.

The baseline is built from the median of an 8-sample Luo reference cluster:

- source: `temperature sweep RMFT pairing data`
- branch: charge-balanced `p ≈ 0`
- temperature window: `temperature_eV <= 1.0e-3`
- samples used: first 8 points sorted by temperature

The resulting baseline channels are:

- `delta_zz_s = 43.47120957876885`
- `delta_zz_d ≈ 0`
- `delta_xx_s = -1.7820360737854513`
- `delta_xx_d ≈ 0`
- `delta_zx_s ≈ 0`
- `delta_zx_d = -3.5075801360800885`
- `delta_perp_z = -63.513372199351885`
- `delta_perp_x = -10.177855352139929`

The weak channel `delta_zx_s` remains numerically tiny in this baseline, which
is consistent with its Stage-3 optional status.

## Current Diagnostics

After regenerating the current round-2 outputs with the Task C freeze gate:

- round-1 median retained ratio total: `0.34670871724588614`
- round-2 median retained ratio total: `0.36607361367472413`
- median retained-ratio improvement: `0.02148692035546879`
- round-1 median residual norm total: `59.27361303524721`
- round-2 median residual norm total: `54.63856967656513`
- median optional/core scale ratio: `0.0` after the default freeze gate
- default optional-channel frozen fraction: `0.98157694582946`

So the repository still keeps a modest round-2 improvement over round 1, while
the weak mixed channel is now explicitly suppressed in the default workflow.

## Truth Layer vs Fit Layer

Stage 3 keeps the repository pointed toward a layered design:

- truth layer:
  the round-2 physical channels with weighted / regularized / gauge-fixed Luo
  projection
- fit layer:
  still to be defined later for surrogate / inverse work as a lower-dimensional
  control space

The key design choice is that fit convenience should not overwrite the physics
truth layer.
