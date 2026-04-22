# LNO327 AR Forward Truth Chain

## Project Goal

This repository is the forward-physics truth-chain project for LNO327 / La3Ni2O7 AR spectroscopy. It provides:

- a reusable physics forward workflow,
- a bridge from Luo's RMFT source data,
- the RMFT-source-to-round-2 projection logic,
- an authoritative formal round-2 baseline,
- and AR-facing validation diagnostics.

Large-scale surrogate training, inverse training, checkpoint management, and
experiment-fitting orchestration are intentionally deferred to a later separate
training / inversion repository.

## Current Status

The repository now has a Stage-3 round-2 pairing truth layer. The active physics
path is:

`Luo source pairing tensors -> PhysicalPairingChannels -> Delta(k) -> interface diagnostics -> multichannel BTK spectrum`

The round-2 truth layer uses seven core channels plus one optional weak channel
(`delta_zx_s`). The formal round-2 baseline is sourced from
`outputs/source/round2_baseline_selection.json`, which is generated from the
low-temperature charge-balanced Luo cluster and read by
`core.presets.base_physical_pairing_channels()`.

The main validation axis is now:

`RMFT source-reference AR spectra -> round-2 projected-channel AR spectra`

See `docs/rmft_source_vs_round2_ar_validation.md` and
`outputs/core/rmft_source_vs_round2_ar_validation/` for the current AR-facing
adequacy check. Round-1 comparisons remain historical / compatibility context
only.

Task H defines the future AR inversion fit layer in
`docs/fit_layer_parameterization_task_h.md`. The fit layer is a conservative
control space around the round-2 truth layer: it reports ranked candidate
families and uncertainty ranges, not a unique microscopic RMFT parameter point.

Task I defines the stable forward interface for a later external training
repository in `docs/forward_interface_task_i.md`. External callers should use
the `forward` package or `scripts/core/generate_forward_spectrum.py` rather than
copying internal normal-state, projection, pairing, interface, or BTK code.

Task L defines the current directional capability contract in
`docs/direction_capability_task_l.md`. In the current forward model,
`interface_angle` is strictly a 2D in-plane interface-normal angle. In-plane
`100` and `110` are supported only as high-symmetry raw-angle shorthand, while
true `c-axis` transport is not yet physically implemented.

Task M promotes those high-symmetry in-plane directions into callable forward
modes, documented in `docs/directional_modes_task_m.md`. External callers may
request `inplane_100` or `inplane_110` through the `forward` directional helpers
without manually supplying raw angles.

Task N validates generic non-high-symmetry in-plane raw angles in
`docs/inplane_generic_direction_validation_task_n.md`. The current result is
that generic raw angles remain diagnostic / caution-required rather than
broadly promoted truth modes.

Task O formally walls off c-axis transport in `docs/c_axis_direction_task_o.md`.
The current truth chain has no `kz`, out-of-plane velocity, or c-axis reflected
state construction, so c-axis must not be emulated with a 2D in-plane
`interface_angle`.

Task P adds a narrow directional-spread primitive in
`docs/directional_spread_task_p.md`. The current supported spread is a uniform
symmetric average around `inplane_100` or `inplane_110` within a half width no
larger than `pi/32`; it is a forward approximation, not an experiment-side
mixture fit.

Task Q publishes the final external directional capability contract for the
current forward interface in `docs/directional_capability_contract_task_q.md`
and `outputs/core/forward_interface/directional_capability_index.json`.
External repositories should treat those files as the compact direction
contract: named `inplane_100` / `inplane_110` modes are supported, generic raw
in-plane angles are diagnostic / caution-only, c-axis is unsupported, and
narrow spread is supported only around named in-plane modes.

The compact index of currently decision-relevant generated outputs is
`docs/current_output_index_task_k.md`.

## Core Principle

The physics forward workflow is the ground-truth path:

`ModelParams -> FS / projection / interface diagnostics -> multichannel BTK spectrum`

Future surrogate or inverse layers must remain downstream approximations of
this workflow and must not replace the forward truth chain in this repository.

## Baseline Provenance

The baseline normal-state parameters remain the repository-local fixed normal
state in `src/core/presets.py`.

The formal round-2 pairing baseline is not handwritten in `presets.py`. It is
read from the authoritative source-side record
`outputs/source/round2_baseline_selection.json`; see
`docs/pairing_state_stage3.md` for the current pairing-state design.
