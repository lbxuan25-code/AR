# Order Parameter Refactor Round 2

## Goal

Round 2 upgrades the local order-parameter language from the restricted
round-1 `PairingParams` ansatz to a source-native physical channel layer while
keeping the normal-state and BTK transport workflow intact.

## Physical Channel Layer

The new formal container is `PhysicalPairingChannels` in
`src/core/parameters.py`. It uses:

- `delta_zz_s`
- `delta_zz_d`
- `delta_xx_s`
- `delta_xx_d`
- `delta_zx_s`
- `delta_zx_d`
- `delta_perp_z`
- `delta_perp_x`

These channels are organized by orbital sector, bond symmetry, and interlayer
structure.

## Source-Native Mapping

Round-2 source projection is implemented in `src/source/round2_projection.py`.
It no longer picks only a few matrix elements. Instead it:

1. reads `delta_x`, `delta_y`, `delta_z` from the Luo source,
2. builds a physically interpretable basis of source-native channel tensors,
3. solves a constrained complex least-squares reconstruction on the full source
   tensor,
4. returns the fitted physical channel parameters plus reconstruction metrics.

This keeps the mapping physical rather than turning it into an opaque numeric
dimensionality reduction.

## Channel -> Delta(k)

`src/core/pairing.py` now supports both the legacy `PairingParams` container and
the round-2 `PhysicalPairingChannels` container.

The BTK interface still only consumes the final `Delta(k)`. The round-2
construction uses:

- `Delta_zz(k) = delta_zz_s * gamma_s(k) + delta_zz_d * gamma_d(k)`
- `Delta_xx(k) = delta_xx_s * gamma_s(k) + delta_xx_d * gamma_d(k)`
- `Delta_zx(k) = delta_zx_s * gamma_s(k) + delta_zx_d * gamma_d(k)`
- `Delta_perp_z(k) = delta_perp_z`
- `Delta_perp_x(k) = delta_perp_x`

so the existing projection / interface / BTK workflow remains usable.

## Why It Is Higher Fidelity Than Round 1

Round 1 only retained a restricted subset of the RMFT tensor. Round 2 expands
the physical language to include:

- z-sector d-like anisotropy,
- x-like interlayer pairing,
- mixed z-x channels.

The new projection is therefore evaluated by reconstruction residual and
retained-ratio metrics rather than by a few hand-picked matrix elements.

On the current Luo source cache, the generated comparison summary reports:

- round-1 median retained ratio total: `0.3467`
- round-2 median retained ratio total: `0.3663`
- median retained-ratio improvement: `0.0217`
- round-1 median residual norm total: `59.27`
- round-2 median residual norm total: `54.49`

So the round-2 channel layer is not just a different parameter grouping. It
produces a measurably better constrained reconstruction of the source pairing
tensors.

## Interface Relation

This refactor does not rewrite the transport solver. It inserts a more faithful
physical representation layer before the same `delta_matrix(k)` ->
`interface_gap_diagnostics(...)` -> `compute_multichannel_btk_conductance(...)`
pipeline.

## Compatibility

The legacy `PairingParams` API still works. `src/core/pairing.py` provides a
compatibility conversion into the round-2 physical channel layer so old round-1
code paths are preserved while round-2 paths can be introduced incrementally.
