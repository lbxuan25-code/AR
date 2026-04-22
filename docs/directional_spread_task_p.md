# Directional Spread Primitive, Task P

## Purpose

Task P adds a minimal forward-level primitive for narrow angular spread around
supported in-plane directional modes. This is meant to represent an idealized
finite angular aperture in the forward truth chain.

It is not an experiment-side mixture fit, not a broad generic-angle training
mode, and not c-axis support.

## Primitive Definition

The primitive is `DirectionalSpread` in the stable `forward` package.

Fields:

| Field | Meaning |
| --- | --- |
| `direction_mode` | Supported named central mode, currently `inplane_100` or `inplane_110` |
| `half_width` | Symmetric angular half width in radians |
| `num_samples` | Odd number of raw-angle samples |
| `averaging_rule` | Currently only `uniform_symmetric` |

Contract:

- sample raw angles are centered on the named mode;
- samples are evenly spaced over `[center - half_width, center + half_width]`;
- weights are uniform;
- `num_samples` must be odd;
- nonzero spread requires at least three samples;
- current maximum half width is `pi/32`;
- c-axis and unsupported generic modes are not allowed.

Mathematically, for samples `theta_i` and weights `w_i = 1/N`:

```text
G_spread(V) = sum_i w_i * G(V; theta_i)
```

The average is applied to normalized spectra produced by the existing 2D
in-plane forward path.

## Public Helpers

Programmatic example:

```python
from forward import (
    BiasGrid,
    DirectionalSpread,
    FitLayerSpectrumRequest,
    generate_spread_spectrum_from_fit_layer,
    transport_with_direction_mode,
)

request = FitLayerSpectrumRequest(
    transport=transport_with_direction_mode("inplane_110", nk=31),
    bias_grid=BiasGrid(bias_min_mev=-40.0, bias_max_mev=40.0, num_bias=161),
)
spread = DirectionalSpread(
    direction_mode="inplane_110",
    half_width=3.141592653589793 / 64.0,
    num_samples=5,
)
result = generate_spread_spectrum_from_fit_layer(request, spread)
```

The source-linked path also has:

```python
generate_spread_spectrum_from_source_round2(request, spread)
```

Spread outputs record:

- the central `direction_mode`;
- the central raw `interface_angle`;
- the spread half width, number of samples, and averaging rule;
- each sampled raw angle and weight;
- per-sample channel diagnostics.

## Validation

Generated artifacts:

- `outputs/core/directional_spread_validation/directional_spread_summary.json`
- `outputs/core/directional_spread_validation/directional_spread_metrics.csv`
- `outputs/core/directional_spread_validation/directional_spread_validation.png`

Validation settings:

- modes: `inplane_100`, `inplane_110`
- half widths: `0`, `pi/128`, `pi/64`, `pi/32`
- barriers: `0.5`, `1.0`
- pairing states: formal baseline and a small fit-layer shifted state
- `num_samples = 5` for nonzero spread
- `nk = 31`
- bias range `[-40, 40]` meV with `161` points

The validation checks width-step smoothness by comparing each spread spectrum
against the previous smaller width for the same mode, barrier, and pairing
state.

Result:

- total cases: `32`
- max observed width-step spectrum difference: `0.20857708180230394`
- threshold: `0.25`
- smoothness verdict: pass

## Safe Use Boundary

Directional spread is currently safe as a narrow forward approximation when:

- the central direction is `inplane_100` or `inplane_110`;
- the half width is no larger than `pi/32`;
- the averaging rule is `uniform_symmetric`;
- generated data retains the spread metadata and sampled angles.

It should not be used to claim:

- true c-axis support;
- broad generic-angle truth-mode support;
- experiment-side fitted direction distributions;
- an arbitrary replacement for Task-N generic-angle validation.

## Final Judgment

The forward truth chain can now generate reproducible narrow spread-averaged
directional spectra for supported in-plane high-symmetry modes. The primitive is
low-dimensional and interpretable, and validation shows smooth spectral
evolution over the documented width range.
