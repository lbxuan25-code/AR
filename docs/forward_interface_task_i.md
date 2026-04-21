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
    FitLayerSpectrumRequest,
    SourceRound2SpectrumRequest,
    TransportControls,
    generate_spectrum_from_fit_layer,
    generate_spectrum_from_source_round2,
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
| `interface_angle` | Interface orientation angle in radians |
| `barrier_z` | BTK barrier strength |
| `gamma` | Broadening in meV |
| `temperature_kelvin` | Temperature in Kelvin |
| `nk` | Momentum-grid resolution used by the interface diagnostics |

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

It can record stable schema and convention metadata without copying the
normal-state, projection, pairing, interface, or BTK implementation.
