# Generic In-Plane Direction Validation, Task N

## Purpose

Task N tests whether the continuous raw `interface_angle` path is reliable
enough to treat generic non-high-symmetry in-plane directions as supported
truth-mode forward requests.

Task N does not add c-axis support and does not add new named public modes.
It validates the existing 2D in-plane raw-angle path from Task L and compares
generic angles against the named high-symmetry modes from Task M.

Generated artifacts:

- `outputs/core/inplane_direction_scan/summary.json`
- `outputs/core/inplane_direction_scan/metrics.csv`
- `outputs/core/inplane_direction_scan/inplane_direction_scan_plots.png`

## Scan Definition

The scan covers `[0, pi/2]` with `33` evenly spaced angles. This includes:

- `0`
- `pi/4`
- `pi/2`
- 30 non-high-symmetry generic in-plane angles

Forward settings:

- formal round-2 baseline pairing
- `nk = 41`
- bias range `[-40, 40]` meV with `201` points
- `barrier_z = 0.5`
- `gamma = 1.0` meV
- `temperature = 3.0` K

For each angle the scan records:

- nominal reflected-state matching;
- tight and loose matching tolerance sensitivity;
- same-band match retention;
- reflected mismatch statistics;
- representative conductance spectrum;
- nearest-neighbor angular spectrum smoothness.

## Tolerance Profiles

| Profile | `k_parallel_tol` | `match_distance_tol` |
| --- | ---: | ---: |
| tight | `0.025` | `0.075` |
| nominal | `0.050` | `0.150` |
| loose | `0.100` | `0.300` |

Tolerance sensitivity is summarized by the span of matched-channel fractions
between these profiles.

## Support Thresholds

Robust support requires all of:

- nominal matched fraction `>= 0.95`
- same-band fraction `>= 0.99`
- nominal p95 reflected mismatch `<= 0.02`
- matched-fraction tolerance span `<= 0.10`
- neighbor max absolute conductance step `<= 0.25`

Caution support requires all of:

- nominal matched fraction `>= 0.35`
- same-band fraction `>= 0.95`
- nominal p95 reflected mismatch `<= 0.18`
- matched-fraction tolerance span `<= 0.70`
- neighbor max absolute conductance step `<= 0.75`

Angles that fail the caution thresholds are classified as unstable.

## Results

Classification counts over all 33 angles:

| Class | Count |
| --- | ---: |
| robust | 3 |
| caution | 20 |
| unstable | 10 |

Classification counts over the 30 non-high-symmetry generic angles:

| Class | Count |
| --- | ---: |
| robust | 0 |
| caution | 20 |
| unstable | 10 |

High-symmetry checkpoints:

| Angle | Class | Nominal matched fraction | p95 mismatch | tolerance span | neighbor max step |
| --- | --- | ---: | ---: | ---: | ---: |
| `0` | robust | 1.000 | `8.88e-16` | 0.000 | 0.213 |
| `pi/4` | robust | 1.000 | `1.06e-15` | 0.000 | 0.216 |
| `pi/2` | robust | 1.000 | `1.08e-15` | 0.000 | 0.213 |

Representative generic checkpoints:

| Angle | Class | Nominal matched fraction | p95 mismatch | tolerance span | neighbor max step |
| --- | --- | ---: | ---: | ---: | ---: |
| `pi/8` | caution | 0.406 | 0.147 | 0.522 | 0.060 |
| `3pi/8` | caution | 0.406 | 0.147 | 0.522 | 0.060 |

Unstable regions appear near several off-symmetry windows. On this 33-angle
grid, unstable points occur in:

- `0.046875π` to `0.093750π`
- `0.171875π`
- `0.328125π`
- `0.406250π` to `0.453125π`

The exact boundaries should be treated as grid-level diagnostics, not as final
physical phase boundaries.

## Decision

Generic non-high-symmetry in-plane raw angles are not promoted to broadly
supported truth modes by Task N.

The current safe truth-mode recommendation remains:

- use `inplane_100` for 2D in-plane `[100]`;
- use `inplane_110` for 2D in-plane `[110]`;
- use raw generic `interface_angle` only as diagnostic / caution-required
  output, with scan metrics recorded alongside generated spectra.

This is a partial support boundary, not a physics proof that all generic
directions are invalid. It says the current reflected-state matching and
angular smoothness diagnostics are not strong enough to certify generic
non-high-symmetry angles as stable forward truth modes.

## Downstream Guidance

External repositories should not train a general generic-angle truth dataset by
blindly sampling raw `interface_angle` over `[0, pi/2]`.

Allowed with current evidence:

- high-symmetry named modes from Task M;
- diagnostic generic-angle scans for method development or sensitivity checks;
- generic-angle examples only if their support class and metrics are retained.

Not allowed as a current truth-mode claim:

- treating every continuous in-plane angle as equally reliable;
- mixing generic raw angles into a training target without recording support
  class, tolerance sensitivity, and mismatch metrics;
- using this task as evidence for c-axis support.

## Next Step

Task O formally marks c-axis transport unsupported in the current forward
truth chain. Task N does not implement or certify c-axis transport.
