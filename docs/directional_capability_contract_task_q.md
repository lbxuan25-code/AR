# Directional Capability Contract For External Repositories

## Purpose

Task Q consolidates the directional work from Tasks L-P into one stable
external contract. A surrogate or inverse repository should use this document
and `outputs/core/forward_interface/directional_capability_index.json` as the
directional truth boundary for the current forward interface.

This contract does not change the physics model. It only states which
directional requests are supported and what provenance every output must carry.

## Angle Convention

`TransportControls.interface_angle` is a 2D in-plane interface-normal angle in
radians. The current model uses the normal vector
`(cos(interface_angle), sin(interface_angle))` in the `kx-ky` plane.

It is not a general 3D crystal-direction selector. In particular, no raw
`interface_angle` value should be relabeled as c-axis transport.

## Supported Direction Requests

| Request type | Status | External use |
| --- | --- | --- |
| `direction_mode="inplane_100"` | supported Tier A | Stable high-symmetry in-plane `[100]` mode with raw angle `0` |
| `direction_mode="inplane_110"` | supported Tier A | Stable high-symmetry in-plane `[110]` mode with raw angle `pi/4` |
| raw generic in-plane angle | diagnostic / caution-only | Computable, but not promoted as a broad external truth mode |
| `direction_mode="c_axis"` and aliases | unsupported | Rejected; current model has no true c-axis transport path |
| `DirectionalSpread(...)` around named modes | supported narrow primitive | Uniform symmetric average around `inplane_100` or `inplane_110` |

The canonical named modes are exposed through `forward.list_directional_modes()`
and `forward.transport_with_direction_mode(...)`.

## Public API

Programmatic named-mode request:

```python
from forward import BiasGrid, FitLayerSpectrumRequest, generate_spectrum_from_fit_layer
from forward import transport_with_direction_mode

request = FitLayerSpectrumRequest(
    pairing_controls={"delta_zz_s": 0.1, "delta_perp_x": -0.1},
    transport=transport_with_direction_mode("inplane_110", nk=21),
    bias_grid=BiasGrid(bias_min_mev=-30.0, bias_max_mev=30.0, num_bias=81),
)
payload = generate_spectrum_from_fit_layer(request).to_dict()
```

Programmatic spread request:

```python
import math

from forward import DirectionalSpread, generate_spread_spectrum_from_fit_layer

spread = DirectionalSpread(
    direction_mode="inplane_110",
    half_width=math.pi / 64.0,
    num_samples=5,
)
payload = generate_spread_spectrum_from_fit_layer(request, spread).to_dict()
```

CLI named-mode request:

```bash
PYTHONPATH=src python scripts/core/generate_forward_spectrum.py fit-layer \
  --direction-mode inplane_110 \
  --control delta_zz_s=0.1 \
  --control delta_perp_x=-0.1 \
  --nk 21 \
  --num-bias 81 \
  --bias-min -30 \
  --bias-max 30 \
  --output-json outputs/core/forward_interface/fit_layer_inplane_110_example_spectrum.json
```

CLI spread request:

```bash
PYTHONPATH=src python scripts/core/generate_forward_spectrum.py fit-layer \
  --direction-mode inplane_110 \
  --spread-half-width 0.04908738521234052 \
  --spread-num-samples 5 \
  --control delta_zz_s=0.1 \
  --control delta_perp_x=-0.1 \
  --nk 21 \
  --num-bias 81 \
  --bias-min -30 \
  --bias-max 30 \
  --output-json outputs/core/forward_interface/fit_layer_inplane_110_spread_example_spectrum.json
```

## Required Output Provenance

Every forward output must serialize:

- `request.transport.direction_mode`
- `request.transport.interface_angle`
- `transport_summary.direction_mode`
- `transport_summary.direction_support_tier`
- `transport_summary.direction_crystal_label`
- `transport_summary.direction_dimensionality`
- `transport_summary.interface_angle`

Spread outputs must additionally serialize:

- `request.directional_spread`
- `metadata.directional_spread`
- `transport_summary.directional_spread`
- `transport_summary.directional_spread_samples`

The `directional_spread_samples` entries record each sampled raw angle,
relative angle, weight, and compact interface matching counts.

## Canonical Example Payloads

Current checked-in examples:

- `outputs/core/forward_interface/fit_layer_example_spectrum.json`
- `outputs/core/forward_interface/source_round2_example_spectrum.json`
- `outputs/core/forward_interface/fit_layer_inplane_110_example_spectrum.json`
- `outputs/core/forward_interface/fit_layer_inplane_110_spread_example_spectrum.json`
- `outputs/core/forward_interface/directional_capability_index.json`

There is no c-axis example because c-axis transport is unsupported.

There is no generic raw-angle example in the stable external set because Task N
classifies generic non-high-symmetry raw angles as diagnostic / caution-only,
not as broad external truth modes.

## External Caller Rules

- Prefer `direction_mode="inplane_100"` or `direction_mode="inplane_110"` over
  manually supplying their raw angles.
- Treat raw non-high-symmetry in-plane angles as diagnostics unless a later task
  promotes a narrower validated subset.
- Do not request or emulate c-axis transport in this repository.
- Use spread only around supported named in-plane modes.
- Keep `half_width <= pi/32` and use odd `num_samples`; nonzero spread requires
  at least 3 samples.
- Preserve the serialized direction provenance fields when building downstream
  datasets.

## Related Evidence

- `docs/direction_capability_task_l.md`: semantic audit of `interface_angle`.
- `docs/directional_modes_task_m.md`: named high-symmetry mode validation.
- `docs/inplane_generic_direction_validation_task_n.md`: generic raw-angle scan.
- `docs/c_axis_direction_task_o.md`: c-axis unsupported decision.
- `docs/directional_spread_task_p.md`: narrow spread validation.
