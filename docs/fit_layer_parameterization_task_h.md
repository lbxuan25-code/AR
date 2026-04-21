# Fit-Layer Parameterization for AR Inversion

## Purpose

Task H defines what an AR inversion should infer from experiment. It does not
change the forward physics implementation, the round-2 truth-layer channel
language, the normal-state model, the source projection, or the formal baseline.

The core separation is:

- truth layer: the repository's current round-2 physical channel layer used to
  generate forward spectra;
- fit layer: a lower-dimensional, regularized control layer used to describe
  which order-parameter features are compatible with AR data.

The fit layer must not claim a unique microscopic RMFT source point. Its output
is a set of confidence-ranked candidate families.

## Truth Layer

The truth layer is the full current round-2 pairing representation:

- container: `PhysicalPairingChannels`
- source projection: `src/source/round2_projection.py`
- formal baseline:
  `outputs/source/round2_baseline_selection.json`
- runtime baseline loader:
  `core.presets.base_physical_pairing_channels()`
- forward path:
  `PhysicalPairingChannels -> Delta(k) -> interface diagnostics -> BTK spectrum`

It contains seven core channels plus one optional weak channel:

- core truth-layer channels:
  `delta_zz_s`, `delta_zz_d`, `delta_xx_s`, `delta_xx_d`,
  `delta_zx_d`, `delta_perp_z`, `delta_perp_x`
- optional weak channel:
  `delta_zx_s`

This is the physics representation the forward repository preserves. The fit
layer below is only a controlled inference interface on top of it.

## Fit-Layer Convention

The default fit layer should be gauge-fixed to the same convention used by the
round-2 source projection and formal baseline. In that gauge, the current
formal baseline is numerically almost real, and `delta_zx_s` is frozen to zero
by the Task-C weak-channel policy.

Therefore the default AR inversion fit should use real-valued channel controls
in meV, interpreted as deviations from or values around the authoritative
formal baseline. Complex phases are not removed from the truth layer, but they
are not default independent fit controls until AR data or source diagnostics
show that phase-sensitive freedom is identifiable.

Recommended default fit vector:

`theta_fit = (pairing_primary, pairing_regularized, weak_channel_policy, transport_nuisance)`

The fit vector controls spectra. It does not define a new microscopic order
parameter language.

## Parameter Table

| Quantity | Truth-layer source | Fit-layer status | Default policy | Reported as |
| --- | --- | --- | --- | --- |
| `delta_zz_s` real amplitude | core round-2 channel | free | primary pairing feature around the formal baseline | inferred feature |
| `delta_xx_s` real amplitude | core round-2 channel | free | primary pairing feature around the formal baseline | inferred feature |
| `delta_zx_d` real amplitude | core round-2 channel | free | primary mixed-channel AR lever; Task-D marked it spectrally visible | inferred feature |
| `delta_perp_z` real amplitude | core round-2 channel | free | primary interlayer z-sector feature | inferred feature |
| `delta_perp_x` real amplitude | core round-2 channel | free | primary interlayer x-sector AR lever; Task-D marked it spectrally visible | inferred feature |
| `delta_zz_d` real amplitude | core round-2 channel | strongly regularized | allowed to vary with a tight prior because the formal baseline is near zero and Task-D found weak leverage | uncertainty band / candidate-family feature |
| `delta_xx_d` real amplitude | core round-2 channel | strongly regularized | allowed to vary with a tight prior because the formal baseline is near zero | uncertainty band / candidate-family feature |
| `delta_zx_s` real amplitude | optional weak channel | fixed by default | default value `0`; open a separate weak-channel branch only if data require it | branch flag plus uncertainty band |
| channel imaginary parts | truth-layer complex channels | fixed by default | set to zero in the projection gauge unless phase freedom is explicitly tested | diagnostic extension, not default fit output |
| global superconducting phase | gauge freedom | fixed | not identifiable from current conductance-only AR spectra | not reported |
| normal-state parameters | `base_normal_state_params()` | fixed | use the repository-local fixed normal state | provenance only |
| formal baseline channels | authoritative baseline record | fixed reference | used as the center / reference state for pairing controls | provenance only |
| interface angle | transport control | free or scan-conditioned | infer when interface geometry is unknown; fix/condition when known experimentally | nuisance parameter |
| barrier strength `barrier_z` | transport control | free | infer jointly with pairing because Task-G worst case appears in low-barrier scans | nuisance parameter |
| broadening `gamma` | transport control | free | infer jointly; captures lifetime / resolution broadening in the current BTK kernel | nuisance parameter |
| temperature | transport control | fixed or narrow-prior | use experimental setpoint when available; otherwise fit with a narrow physical prior | nuisance parameter |
| spectrum normalization | experimental preprocessing | derived / external | keep outside the physics truth layer; record any normalization used by the fitting repository | metadata / nuisance |
| peak positions and zero-bias conductance | AR spectrum | derived | compute from generated spectra, do not fit as independent physics parameters | observables |

## Default Free Set

The recommended default free pairing controls are:

- `delta_zz_s`
- `delta_xx_s`
- `delta_zx_d`
- `delta_perp_z`
- `delta_perp_x`

These are the main nonzero, AR-visible controls around the formal baseline.

The recommended strongly regularized pairing controls are:

- `delta_zz_d`
- `delta_xx_d`

They remain part of the truth layer, but the default inversion should not let
them absorb noise without evidence because the current formal baseline values
are close to zero.

The recommended weak optional branch is:

- `delta_zx_s = 0` in the default fit family;
- optional nonzero `delta_zx_s` branch only when it produces a clearly better
  AR fit that cannot be reproduced by the core controls and transport nuisance
  parameters.

## Transport Policy

Transport parameters are not order-parameter features. They are nuisance
controls that must be inferred or conditioned so pairing features are not
overclaimed.

Default policy:

- infer `barrier_z` and `gamma` jointly with pairing;
- infer `interface_angle` when geometry is unknown, otherwise condition on the
  known interface orientation;
- fix `temperature` to the experimental setpoint when available, otherwise use
  a narrow prior.

The inversion report must separate pairing-feature uncertainty from transport
nuisance uncertainty.

## Inversion Output Contract

The inversion output should be a ranked set of candidate families, not one
claimed true microscopic parameter point.

Each reported family should include:

- fit-family identifier;
- pairing controls with uncertainties or accepted ranges;
- transport nuisance controls with uncertainties or accepted ranges;
- whether the weak `delta_zx_s` branch is active;
- generated AR spectra and residual metrics;
- confidence rank or relative objective score;
- notes on degeneracies, especially pairing-vs-transport tradeoffs.

The report should avoid saying "the order parameter is uniquely X." It should
say "the AR data are compatible with these feature families under the current
forward model."

## What Exactly Will Be Inferred

The fit layer infers AR-visible order-parameter features:

- relative strength of z-like, x-like, mixed, and interlayer pairing sectors;
- whether d-like corrections are needed beyond tight priors;
- whether the optional `delta_zx_s` branch is required;
- transport nuisance conditions needed to reproduce the spectrum.

It does not infer:

- a unique RMFT source sample;
- a full source tensor;
- a new normal-state Hamiltonian;
- a unique microscopic ground state;
- surrogate or inverse-training internals.

## Task-G Connection

Task G found that the current round-2 truth layer is sufficient for
representative AR forward spectra against source-reference tensors, while
low-barrier cases still show the largest local discrepancies. That supports a
fit layer based on round-2 channels, but it also argues for conservative
reporting:

- keep source-reference AR fidelity as the validation anchor;
- track low-barrier cases carefully;
- treat weak or near-zero channels as uncertainty-family features, not as
  unconstrained fit knobs;
- keep truth-layer code separate from future training / inversion machinery.

## Forward-Interface Use

Task I exposes this fit-layer convention through the stable forward interface
documented in `docs/forward_interface_task_i.md`.

External callers should use:

- `forward.generate_spectrum_from_fit_layer(...)` for Task-H controls;
- `forward.generate_spectrum_from_source_round2(...)` for Luo samples projected
  into the round-2 truth layer.

This keeps the fit layer as an inference contract while preserving the current
forward repository as the source of truth for spectra.
