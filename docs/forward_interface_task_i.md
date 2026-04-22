# Stable Forward Interface for External Training Repositories

## Purpose

Task I prepares this repository to act as a stable forward-physics dependency
for a later separate surrogate / inverse repository.

This task does not create the training repository. It defines how an external
repository can call the current forward engine without copying internal physics
code.

## Public Entry Points

The stable callable package is `forward`.

Programmatic imports:

```python
from forward import (
    BiasGrid,
    DirectionalSpread,
    FitLayerSpectrumRequest,
    SourceRound2SpectrumRequest,
    TransportControls,
    generate_spectrum_from_fit_layer,
    generate_spectrum_from_source_round2,
    generate_spread_spectrum_from_fit_layer,
    generate_spread_spectrum_from_source_round2,
    transport_with_direction_mode,
)
```

Command-line entry point:

```bash
PYTHONPATH=src python scripts/core/generate_forward_spectrum.py fit-layer \
  --control delta_zz_s=0.1 \
  --control delta_perp_x=-0.1 \
  --nk 21 \
  --num-bias 81 \
  --bias-min -30 \
  --bias-max 30 \
  --output-json outputs/core/forward_interface/fit_layer_example_spectrum.json
```

```bash
PYTHONPATH=src python scripts/core/generate_forward_spectrum.py source-round2 \
  --sample-index 0 \
  --nk 21 \
  --num-bias 81 \
  --bias-min -30 \
  --bias-max 30 \
  --output-json outputs/core/forward_interface/source_round2_example_spectrum.json
```

The generated examples are:

- `outputs/core/forward_interface/fit_layer_example_spectrum.json`
- `outputs/core/forward_interface/source_round2_example_spectrum.json`
- `outputs/core/forward_interface/fit_layer_inplane_110_example_spectrum.json`
- `outputs/core/forward_interface/fit_layer_inplane_110_spread_example_spectrum.json`
- `outputs/core/forward_interface/directional_capability_index.json`

## Interface Version Tags

The current interface constants live in `src/forward/schema.py`.

| Field | Current value | Meaning |
| --- | --- | --- |
| `FORWARD_INTERFACE_VERSION` | `ar_forward_v1` | Stable public forward API identity |
| `FORWARD_INPUT_SCHEMA_VERSION` | `ar_forward_input_v1` | Canonical request schema identity |
| `FORWARD_OUTPUT_SCHEMA_VERSION` | `ar_forward_output_v1` | Canonical response schema identity |
| `ROUND2_PAIRING_CONVENTION_ID` | `round2_physical_channels_task_h_fit_layer_v1` | Pairing / fit-layer convention identity |

External datasets should record these fields from each output payload. If any
of these values changes later, downstream training data should treat the new
spectra as a distinct forward definition.

## Canonical Input Schemas

### Fit-Layer Request

Class: `FitLayerSpectrumRequest`

Fields:

| Field | Meaning |
| --- | --- |
| `pairing_controls` | Real meV fit-layer controls keyed by Task-H channel name |
| `pairing_control_mode` | `delta_from_baseline_meV` or `absolute_meV` |
| `allow_weak_delta_zx_s` | Opens the optional weak-channel branch when true |
| `transport` | `TransportControls` |
| `bias_grid` | `BiasGrid` |
| `request_label` | Free label copied into output metadata |

Default pairing behavior:

- controls are interpreted as deviations from the authoritative formal
  baseline;
- unspecified channels keep their formal baseline value;
- `delta_zx_s` stays fixed to zero unless `allow_weak_delta_zx_s=True`;
- controls are real in the projection gauge from Task H.

### Source-Linked Round-2 Request

Class: `SourceRound2SpectrumRequest`

Fields:

| Field | Meaning |
| --- | --- |
| `source_sample_id` | Exact Luo sample id to project |
| `source_sample_index` | Alternative integer index into `load_luo_samples()` |
| `transport` | `TransportControls` |
| `bias_grid` | `BiasGrid` |
| `request_label` | Free label copied into output metadata |

Exactly one of `source_sample_id` or `source_sample_index` should be supplied.
The sample is projected through the default round-2 source projection before
generating the AR spectrum.

### Transport and Bias

`TransportControls` fields:

| Field | Meaning |
| --- | --- |
| `direction_mode` | Optional named in-plane high-symmetry mode provenance |
| `interface_angle` | 2D in-plane interface-normal angle in radians |
| `barrier_z` | BTK barrier strength |
| `gamma` | Broadening in meV |
| `temperature_kelvin` | Temperature in Kelvin |
| `nk` | Momentum-grid resolution used by the interface diagnostics |

Direction convention:

- `interface_angle` is not a general 3D experimental crystal-direction selector.
- It is the polar angle of the interface normal in the current model's 2D
  `kx-ky` plane, with normal vector `(cos(angle), sin(angle))`.
- In-plane `100` is currently valid only as shorthand for raw
  `interface_angle = 0` or symmetry-equivalent `pi/2`.
- In-plane `110` is currently valid only as shorthand for raw
  `interface_angle = pi/4`.
- True `c-axis` transport is not implemented in the current forward model.

See `docs/direction_capability_task_l.md` for the Task-L audit and support
tiers.
See `docs/c_axis_direction_task_o.md` for the formal Task-O c-axis unsupported
decision and required extension plan.

Task M adds named helpers for the Tier-A high-symmetry in-plane modes:

| Direction mode | Raw `interface_angle` | Meaning |
| --- | ---: | --- |
| `inplane_100` | `0` | 2D in-plane `[100]` interface normal |
| `inplane_110` | `pi/4` | 2D in-plane `[110]` interface normal |

Programmatic callers should prefer:

```python
transport = transport_with_direction_mode("inplane_110", nk=41)
```

The forward engine validates that `direction_mode` and `interface_angle` are
consistent. Named-mode outputs record both the mode label and the raw angle in
the serialized request and `transport_summary`.

The CLI equivalent is:

```bash
PYTHONPATH=src python scripts/core/generate_forward_spectrum.py fit-layer \
  --direction-mode inplane_110 \
  --output-json outputs/core/forward_interface/example_inplane_110.json
```

Task N validates generic non-high-symmetry raw angles. The current support
boundary is conservative: generic raw `interface_angle` values are computable,
but they are diagnostic / caution-required unless their scan metrics satisfy
the robust thresholds in `docs/inplane_generic_direction_validation_task_n.md`.
External training repositories should not treat arbitrary continuous in-plane
angles as broadly supported truth modes.

Task O formally forbids c-axis labeling in the current public interface.
`direction_mode="c_axis"` and common c-axis aliases are rejected because the
current model has no `kz`, no out-of-plane velocity, and no c-axis reflected
state construction. A raw 2D `interface_angle` output must never be relabeled as
c-axis transport.

Task P adds narrow spread helpers for supported in-plane modes:

```python
from forward import DirectionalSpread, generate_spread_spectrum_from_fit_layer

spread = DirectionalSpread(
    direction_mode="inplane_110",
    half_width=3.141592653589793 / 64.0,
    num_samples=5,
)
result = generate_spread_spectrum_from_fit_layer(request, spread)
```

The spread rule is a uniform symmetric average over raw-angle spectra around
the central named mode. The current half-width contract is narrow,
`half_width <= pi/32`, and spread outputs record the spread settings plus each
sample angle and weight. This is a forward approximation only, not an
experiment-side fitted directional mixture.

Task Q consolidates Tasks L-P into the final external directional contract for
the current interface. External repositories should use
`docs/directional_capability_contract_task_q.md` plus the compact
machine-readable index
`outputs/core/forward_interface/directional_capability_index.json` when deciding
which direction requests are safe to generate.

`BiasGrid` fields:

| Field | Meaning |
| --- | --- |
| `bias_min_mev` | Minimum bias in meV |
| `bias_max_mev` | Maximum bias in meV |
| `num_bias` | Number of bias points |

## Canonical Output Schema

Class: `ForwardSpectrumResult`

Top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Output schema version |
| `request_kind` | `fit_layer` or `source_round2` |
| `request` | Serialized canonical input request |
| `metadata` | Versioned provenance and convention metadata |
| `pairing_channels` | Final round-2 physical channels used by the forward model |
| `bias_mev` | Bias grid in meV |
| `conductance` | Broadened normalized AR conductance |
| `conductance_unbroadened` | Conductance before thermal / broadening post-processing |
| `transport_summary` | Compact BTK / interface diagnostic summary |

Complex channels are serialized as:

```json
{"re": 1.0, "im": 0.0}
```

## Canonical Metadata Fields

Every output includes:

- `forward_interface_version`
- `output_schema_version`
- `pairing_convention_id`
- `pairing_source`
- `normal_state_policy`
- `normal_state_family`
- `truth_layer`
- `fit_layer_policy`
- `projection_config`
- `formal_baseline_record`
- `formal_baseline_role`
- `formal_baseline_selection_rule`
- `weak_channel_policy`
- `git_commit`
- `git_dirty`

Direction provenance is serialized in every output under `request.transport`
and `transport_summary`:

- `direction_mode`
- `interface_angle`
- `direction_support_tier`
- `direction_crystal_label`
- `direction_dimensionality`

Directional-spread outputs additionally include:

- `request.directional_spread`
- `metadata.directional_spread`
- `transport_summary.directional_spread`
- `transport_summary.directional_spread_samples`

`formal_baseline_record` is stored as the repository-relative path
`outputs/source/round2_baseline_selection.json`, not as a machine-local
absolute path.

The checked-in example payloads under `outputs/core/forward_interface/` are
canonical clean-snapshot examples and should have `git_dirty = false`.

Source-linked requests additionally include:

- `source_sample_id`
- `source_sample_kind`
- `source_coordinates`
- `round2_projection_metrics`
- `round2_projection_metadata`

These metadata fields are the minimum provenance an external training dataset
should store with each generated spectrum.

## Authoritative Conventions

The stable interface freezes the current conventions used by this forward
repository:

- normal-state policy:
  `base_normal_state_params()` from `src/core/presets.py`
- formal pairing baseline:
  `outputs/source/round2_baseline_selection.json`
- truth-layer container:
  `PhysicalPairingChannels`
- default source projection:
  `src/source/round2_projection.py`
- fit-layer convention:
  `docs/fit_layer_parameterization_task_h.md`
- AR validation anchor:
  `docs/rmft_source_vs_round2_ar_validation.md`

The interface intentionally does not expose arbitrary normal-state edits,
surrogate models, inverse optimizers, or training loops.

## Acceptance Check

A separate training repository can now depend on this repository by importing
`forward` or by calling `scripts/core/generate_forward_spectrum.py`. It can
generate spectra from:

- Task-H fit-layer controls;
- Luo source samples projected into round-2 channels.
- supported named in-plane directions and narrow directional spreads described
  by `docs/directional_capability_contract_task_q.md`.

It can record stable schema and convention metadata without copying the
normal-state, projection, pairing, interface, or BTK implementation.
