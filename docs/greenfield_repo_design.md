# Greenfield Repo Design

## Goal

This repository follows the `AGENTS_1.md` requirement to build a minimal closed
loop:

`physics core -> source bridge -> dataset -> surrogate -> inverse demo`

## Physics Core Strategy

- Strategy used: selective migration from the local `LNO327_AR_Phenomenology`
  repository.
- Reason: this matches the preferred round-1 path in `AGENTS_1.md` and avoids
  rewriting both the physics core and surrogate simultaneously.

## Baseline Provenance

- `base_normal_state_params()` is migrated from the local
  `LNO327_AR_Phenomenology/src/presets.py`.
- `base_pairing_params()` is migrated from the same baseline preset file.
- The migrated baseline keeps the old repository values and documents this
  explicitly instead of introducing placeholder parameters.

## Round-1 Boundary

- Normal-state parameters stay fixed in round 1.
- Surrogate training space includes pairing raw parameters and transport
  parameters only.
- Inverse outputs are top-K candidate clusters, not a unique true solution.

