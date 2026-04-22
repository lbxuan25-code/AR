# Directional Modes, Task M

## Purpose

Task M promotes the Task-L high-symmetry in-plane raw angles into canonical,
named forward modes. External callers no longer need to remember which raw
`interface_angle` corresponds to in-plane `100` or `110` transport.

This task does not certify generic non-high-symmetry in-plane angles and does
not implement c-axis transport.

## Canonical Modes

| Direction mode | Crystal shorthand | Raw `interface_angle` | Support tier | Meaning |
| --- | --- | ---: | --- | --- |
| `inplane_100` | `100` | `0` | A | 2D in-plane `[100]` interface normal |
| `inplane_110` | `110` | `pi/4` | A | 2D in-plane `[110]` interface normal |

These are named wrappers around the existing 2D in-plane interface geometry.
They do not change the physics path:

`direction_mode -> raw interface_angle -> interface diagnostics -> BTK spectrum`

## Public API

The stable `forward` package now exports:

```python
from forward import (
    interface_angle_for_direction_mode,
    list_directional_modes,
    replace_direction_mode,
    transport_with_direction_mode,
)
```

Example:

```python
from forward import BiasGrid, FitLayerSpectrumRequest, generate_spectrum_from_fit_layer
from forward import transport_with_direction_mode

request = FitLayerSpectrumRequest(
    transport=transport_with_direction_mode("inplane_110", nk=41),
    bias_grid=BiasGrid(bias_min_mev=-40.0, bias_max_mev=40.0, num_bias=201),
)
result = generate_spectrum_from_fit_layer(request)
```

The serialized request records both:

- `transport.direction_mode`
- `transport.interface_angle`

The output `transport_summary` also records:

- `direction_mode`
- `direction_crystal_label`
- `direction_support_tier`
- `direction_dimensionality`
- `interface_angle`

If a caller manually constructs inconsistent transport controls, for example
`direction_mode="inplane_110"` with `interface_angle=0`, the forward engine
raises a `ValueError` instead of silently choosing one meaning.

## CLI Usage

The forward spectrum script accepts named modes:

```bash
PYTHONPATH=src python scripts/core/generate_forward_spectrum.py fit-layer \
  --direction-mode inplane_110 \
  --nk 41 \
  --num-bias 201 \
  --output-json outputs/core/forward_interface/example_inplane_110.json
```

When `--direction-mode` is supplied, it overrides `--interface-angle` using the
canonical mapping.

## Validation

Generated artifacts:

- `outputs/core/directional_modes_validation/directional_modes_summary.json`
- `outputs/core/directional_modes_validation/directional_modes_metrics.csv`
- `outputs/core/directional_modes_validation/directional_modes_comparison.png`

Validation settings:

- formal round-2 baseline through the Task-H fit-layer request path
- `nk = 41`
- bias range `[-40, 40]` meV with `201` points
- `barrier_z = 0.5`
- `gamma = 1.0` meV
- `temperature = 3.0` K

Comparison results:

| Direction mode | Raw angle | Named angle | max abs spectrum diff | raw channels | named channels | same-band channels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `inplane_100` | `0` | `0` | `0.0` | 131 | 131 | 131 |
| `inplane_110` | `pi/4` | `pi/4` | `0.0` | 133 | 133 | 133 |

The named modes reproduce their raw-angle calls exactly in this validation
because they are explicit, checked wrappers around the same forward path.

## Boundary

Supported now:

- `inplane_100`
- `inplane_110`

Still not promoted by Task M:

- generic non-high-symmetry in-plane modes, deferred to Task N;
- true c-axis transport, formally marked unsupported by Task O;
- directional spread, added later by Task P as a narrow uniform symmetric
  forward approximation.

## Final Judgment

Task M establishes canonical callable high-symmetry in-plane modes for external
repositories. External callers can request `100` and `110` transport through
`inplane_100` and `inplane_110`, and outputs now preserve both the named mode
and the raw angle used by the current 2D forward model.
