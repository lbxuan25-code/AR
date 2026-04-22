# C-Axis Direction Audit, Task O

## Purpose

Task O resolves the current ambiguity around `c-axis` transport. The outcome is
explicit:

> True c-axis transport is not supported by the current forward truth chain.

The current model remains a 2D in-plane `kx-ky` interface model. No raw
`interface_angle` value may be relabeled as c-axis transport.

Generated audit artifacts:

- `outputs/core/c_axis_direction_audit/summary.json`
- `outputs/core/c_axis_direction_audit/capability_matrix.csv`

## Audit Result

The current repository does not have enough dimensional structure to implement
true c-axis injection.

| Component | Current evidence | Missing for c-axis |
| --- | --- | --- |
| `normal_state.h0_matrix` | signature is `(kx, ky, params)` | no `kz` or equivalent out-of-plane coordinate |
| `pairing.delta_matrix` | signature is `(kx, ky, params)` | no pairing evaluation on c-axis-injected states |
| `bdg.bdg_matrix` | signature is `(kx, ky, params)` | no 3D BdG momentum structure |
| `SimulationModel` | public builders accept only `kx, ky` | no public 3D/c-axis model API |
| Fermi-surface extraction | uses a 2D square `kx-ky` grid | no 3D Fermi surface or c-axis injection manifold |
| Interface normal | `(cos(angle), sin(angle))` | no out-of-plane normal vector |
| Group velocity | finite differences in `kx` and `ky` only | no `v_z` |

These are blocking gaps, not cosmetic omissions. A c-axis mode would require a
new physical representation, not just a different `interface_angle`.

## Public Interface Policy

Current allowed named direction modes:

- `inplane_100`
- `inplane_110`

Forbidden:

- `c_axis`
- `c-axis`
- `caxis`
- `axis_c`

The `forward` direction registry now gives a specific error for c-axis aliases:

```text
c-axis transport is not supported by the current forward model.
Do not emulate c-axis by a 2D in-plane interface_angle.
```

The CLI cannot request c-axis because `--direction-mode` is restricted to the
supported in-plane modes.

## Required Extension Plan

Before c-axis can become a forward truth mode, the repository would need:

1. A microscopic or phenomenological out-of-plane momentum coordinate, such as
   `kz`, or an explicitly defined equivalent c-axis injection manifold.
2. A normal-state Hamiltonian and pairing matrix that can be evaluated on that
   coordinate.
3. A c-axis transport-state construction that does not reuse the current 2D
   `kx-ky` Fermi contours as if they were c-axis states.
4. A c-axis interface normal, reflected-state construction, and velocity
   projection with an out-of-plane velocity component.
5. Dedicated c-axis reflection diagnostics and spectra before exposing any
   public `c_axis` mode.

## Final Judgment

The current forward truth chain cannot represent true c-axis transport.
Therefore c-axis is formally unsupported and must not be emulated by any raw 2D
in-plane `interface_angle`.

Downstream experiment-fitting code must treat c-axis as unavailable until a
future task explicitly implements and validates a real c-axis path.
