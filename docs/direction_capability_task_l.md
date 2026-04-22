# Direction Capability Audit, Task L

## Background

Task L fixes the directional semantics of the current forward truth chain before
any downstream repository treats `interface_angle` as a general experimental
crystal-direction selector.

The current code path exposes a continuous `interface_angle`, while project
language has also used shorthand such as `100`, `110`, and `c-axis`. This audit
separates what is actually implemented from what is only shorthand or not yet
physically represented.

Generated audit artifacts:

- `outputs/core/direction_capability_audit/direction_capability_summary.json`
- `outputs/core/direction_capability_audit/direction_capability_metrics.csv`
- `outputs/core/direction_capability_audit/direction_capability_representative_plots.png`

## Current `interface_angle` Semantics

`interface_angle` is a 2D in-plane polar angle, in radians, for the interface
normal in the model `kx-ky` plane.

The geometry implemented in `src/core/interface_geometry.py` is:

```text
normal(interface_angle)  = (cos(interface_angle), sin(interface_angle))
tangent(interface_angle) = (-sin(interface_angle), cos(interface_angle))
k_out = k_in - 2 * (k_in dot normal) * normal
```

The reflected momentum is then wrapped back into the square Brillouin zone.
Incident states are selected by negative interface-normal velocity, reflected
states are matched on the existing 2D Fermi-surface contours, and the BTK path
uses the matched 2D interface diagnostics.

This is strictly a 2D in-plane interface-normal convention. The current forward
model does not implement a general 3D crystal-direction interface.

## Direction Support Status

| Label | Current status | Meaning in this repository |
| --- | --- | --- |
| `100` | Supported only as in-plane shorthand | Raw `interface_angle = 0` or symmetry-equivalent `pi/2` |
| `110` | Supported only as in-plane shorthand | Raw `interface_angle = pi/4` |
| generic in-plane angle | Computable with caution | Continuous raw angle in the 2D `kx-ky` plane; broad reliability awaits Task N |
| `c-axis` | Not supported | No `kz`, out-of-plane velocity, or c-axis injection path exists in the current interface geometry |

Important nuance: `100` and `110` are not yet named public forward modes. They
are currently safe shorthand for raw in-plane angles only. Task M may promote
them into canonical named directional modes.

## Representative Scan

The Task-L audit scanned the formal round-2 baseline at:

- `0`
- `pi/8`
- `pi/4`
- `3pi/8`
- `pi/2`

Transport settings:

- `nk = 41`
- bias range `[-40, 40]` meV with `201` points
- `barrier_z = 0.5`
- `gamma = 1.0` meV
- `temperature = 3.0` K

Summary metrics:

| Angle label | Angle | Tier | Incident | Matched | Matched fraction | Same-band fraction | mismatch p95 | Filtered fraction |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `inplane_100_x_axis` | `0` | A | 131 | 131 | 1.000 | 1.000 | `8.88e-16` | 0.000 |
| `inplane_generic_pi_over_8` | `pi/8` | B | 69 | 28 | 0.406 | 1.000 | `1.47e-01` | 0.000 |
| `inplane_110_diagonal` | `pi/4` | A | 133 | 133 | 1.000 | 1.000 | `1.06e-15` | 0.000 |
| `inplane_generic_3pi_over_8` | `3pi/8` | B | 69 | 28 | 0.406 | 1.000 | `1.47e-01` | 0.000 |
| `inplane_100_y_axis_equivalent` | `pi/2` | A | 131 | 131 | 1.000 | 1.000 | `1.08e-15` | 0.000 |

The high-symmetry in-plane directions have exact same-band matching in this
scan and reflected-state mismatch near machine precision. The two generic
angles are computable and produce representative spectra, but many incident
channels do not receive a reflected contour match under the current local
matching tolerances. This is why they remain caution-required rather than fully
promoted truth modes.

## Reliability Tiers

Tier A: reliable and physically supported in the current model.

This includes in-plane high-symmetry raw-angle calls:

- `100` shorthand: `interface_angle = 0` or `pi/2`
- `110` shorthand: `interface_angle = pi/4`

Tier B: computable but approximate / caution-required.

This includes generic continuous in-plane raw angles. They are mathematically
accepted by the 2D geometry and can generate spectra, but the representative
Task-L scan shows reduced reflected-state matching away from high-symmetry
angles. Dense validation and support thresholds are deferred to Task N.

Tier C: not yet physically supported.

This includes `c-axis`. The current model is a 2D `kx-ky` interface-normal
construction and has no true out-of-plane injection coordinate, no `kz`
Fermi-surface matching, and no c-axis velocity/reflection path.

## Public Interface Consequence

External callers may use `TransportControls.interface_angle` only as a raw 2D
in-plane interface-normal angle. They must not interpret it as a general
experimental direction selector.

Safe current use:

- call raw high-symmetry in-plane angles directly when the intended geometry is
  `100` or `110` in the current 2D model;
- record the raw angle and this Task-L capability document with generated data.

Unsafe current use:

- label a raw `interface_angle` result as true `c-axis`;
- mix 2D in-plane and out-of-plane experimental directions as if both were
  supported by the same current forward path;
- assume all generic in-plane angles are equally robust truth modes before
  Task N.

## Final Judgment

The current forward truth chain supports a 2D in-plane raw-angle interface path.
In-plane `100` and `110` are valid shorthand for high-symmetry raw angles in
that 2D convention. Generic in-plane angles are computable but require caution.
True `c-axis` transport is not physically implemented.

Therefore, Task L does not justify any c-axis training or experiment-fitting
claim from this repository. Task O later formally walls off c-axis transport as
unsupported in the current model.

## Next Step

Task M may now promote in-plane high-symmetry raw angles into explicit named
forward modes such as `inplane_100` and `inplane_110`. It should not implement
generic-angle certification or c-axis support; those are separate backlog tasks.
