# TODO

## Current Task

### Task E — Sync documentation with the actual Stage-3 implementation
Bring docs into exact agreement with the current repository state.

#### Update docs to reflect
- 7 core channels + 1 optional weak channel
- weighted ridge + global gauge fix projection
- formal round-2 baseline from the low-temperature charge-balanced Luo cluster
- unified projection metrics
- current limitation: better than round-1, but still not full-RMFT-equivalent

#### Acceptance
A new developer or Codex can read the docs and understand the current pairing-state design without reverse-engineering the code.

---

## Backlog

### Task F — Workspace cleanup / decontamination
Remove obsolete modified content so no stale version can pollute future work.

#### Hard rule
After cleanup, the workspace must contain **one authoritative implementation path** for the current round-2 truth layer.

#### Implement
- Delete or rewrite superseded files from earlier iterations if they are no longer part of the current path
- Remove duplicate scripts / docs / outputs that describe older behavior
- Remove temporary files, backups, scratch notebooks, debug scripts, and obsolete generated artifacts
- In `outputs/`, keep only artifacts that are:
  - produced by the current implementation
  - referenced by current docs/tests
  - or explicitly needed as current-stage diagnostics
- In `docs/`, keep only documents matching the current implementation
- Keep compatibility code only if it is still required by current workflow/tests

#### Deliverables
- cleanup summary listing:
  - deleted files
  - rewritten files
  - intentionally preserved compatibility files and why they remain

#### Acceptance
No stale modified content remains that could plausibly mislead future coding decisions.

---

## Archive

### Task D — Spectral validation of the formal round-2 baseline
Completed.

#### Goal
Verify what the formal round-2 truth-layer state changes in actual AR spectra.

#### Completed items
- added a dedicated spectral-validation diagnostic for:
  - the legacy-compatible baseline
  - the formal round-2 baseline
  - representative Luo projected samples
- scanned the four transport controls:
  - `interface_angle`
  - `barrier_z`
  - `gamma`
  - `temperature`
- added formal-baseline channel-ablation checks for:
  - `delta_zz_d`
  - `delta_zx_d`
  - `delta_perp_x`
- generated side-by-side spectrum comparison plots plus per-scan quantitative metrics
- established a quantitative verdict for what the formal round-2 baseline adds beyond the compatibility baseline

#### Deliverables
- `outputs/core/round2_baseline_spectral_validation/round2_baseline_spectral_validation_summary.json`
- `outputs/core/round2_baseline_spectral_validation/round2_baseline_spectral_validation_metrics.csv`
- `outputs/core/round2_baseline_spectral_validation/round2_baseline_scan_comparison.png`
- `outputs/core/round2_baseline_spectral_validation/round2_channel_sensitivity_scan.png`

#### Result
The repository now has a verified spectral audit showing that the formal round-2 baseline changes AR spectra relative to the legacy-compatible baseline, with `delta_zx_d` and `delta_perp_x` carrying visible spectral leverage while `delta_zz_d` is negligible at the current baseline amplitude.

### Task C — Freeze the weak optional channel by default
Completed.

#### Goal
Treat `delta_zx_s` as a weak optional channel in the default truth-layer workflow.

#### Completed items
- kept `delta_zx_s` fully supported in the round-2 channel language
- added a default soft-freeze gate in the projection workflow:
  - compute the full fit including `delta_zx_s`
  - compute the refit with `delta_zx_s` frozen to zero
  - only activate `delta_zx_s` by default if it clears a clear-need threshold in both:
    - relative magnitude vs the strongest core channel
    - residual-norm improvement vs the frozen refit
- recorded the optional-channel decision in projection metadata and summary diagnostics
- updated the current truth-layer doc to make the weak-channel policy explicit
- regenerated the round-2 summary outputs under the new default policy

#### Deliverables
- updated `src/source/round2_projection.py`
- updated `src/source/round2_projection_diagnostics.py`
- updated `docs/pairing_state_stage3.md`
- updated `outputs/source/round2_projection_summary.json`
- updated `outputs/source/round1_vs_round2_projection_comparison.json`

#### Result
The default truth model is now explicitly centered on the 7 core channels. On the current Luo source cache, `delta_zx_s` has median `0`, p95 `0`, and baseline value `0`, while still remaining available for non-default diagnostic runs.

### Task B — Make the projection AR-aware
Completed.

#### Goal
Upgrade the current weighted-ridge projection so it can emphasize source components that matter most for final AR spectra.

#### Completed items
- added an optional AR-aware source-entry weighting mode on top of the existing block-weight + ridge + gauge-fix projection
- derived AR relevance scores from interface-gap diagnostics of unit round-2 channels on the baseline normal state
- kept the default weighted-ridge path intact while making the AR-aware path explicitly configurable
- built a comparison diagnostic that measures:
  - retained ratio
  - residual norm
  - projected-channel stability
  - representative-sample BTK spectral agreement against a source-tensor reference model
- explicitly tested whether AR-aware weighting helps relative to the current default

#### Deliverables
- `outputs/source/ar_aware_projection_comparison_summary.json`
- `outputs/source/ar_aware_projection_examples.csv`
- `outputs/source/ar_aware_projection_representative_spectra.png`

#### Result
AR-aware entry weighting was explicitly shown not to materially help relative to the current default weighted-ridge projection path, so the repository keeps the existing default as the authoritative projection baseline.

### Task A — Residual anatomy audit
Completed.

#### Goal
Find out exactly what source information is still not captured by the current round-2 pairing truth layer.

#### Completed items
- added round-2 residual anatomy diagnostics with:
  - `delta_x`, `delta_y`, `delta_z` block summaries
  - matrix-entry residual hotspot tables
  - channel-group summaries (`zz`, `xx`, `zx`, `perp`, plus residual `other`)
- saved representative best / median / worst samples
- generated residual heatmap outputs for aggregate and representative cases
- added a short docs note explaining the dominant residual pattern
- determined that the remaining mismatch is more likely dominated by missing channel structure than by the current projection weighting

#### Deliverables
- `outputs/source/round2_residual_anatomy_summary.json`
- `outputs/source/round2_residual_examples.csv`
- `outputs/source/round2_residual_anatomy_heatmaps.png`
- `outputs/source/round2_residual_representatives.png`
- `docs/round2_residual_anatomy.md`

#### Result
The repository now has a verified residual-anatomy audit for the current round-2 truth layer, and future projection work can target the actual residual hotspots rather than guessing broadly.

### Stage_1 — Independent AR physics repository
Completed.

#### Goal
Build a clean standalone repository for the LNO327 AR project, separate from the old workflow.

#### Completed items
- established the new repository structure around:
  - physics core
  - source bridge
  - dataset builder
  - surrogate training path
  - surrogate-assisted inverse path
- implemented the baseline normal-state + pairing + BTK forward workflow
- ensured the repository can produce AR spectra from `ModelParams`
- established the rule that physics forward is the truth chain; surrogate is only an accelerator

#### Result
The project has a functioning standalone physics-forward backbone.

---

### Stage_2 — Luo source bridge and round-1 projection audit
Completed.

#### Goal
Connect Luo RMFT data to the new repository and verify the meaning of the round-1 source projection.

#### Completed items
- inspected Luo source structure and identified the relevant RMFT observables
- established the source language:
  - `delta_x`
  - `delta_y`
  - `delta_z`
- checked basis / semantics / units consistency between Luo source and local repository
- implemented and audited round-1 projection
- proved that round-1 retained-channel formulas were implemented correctly
- proved that round-1 was only a restricted approximation, not a full RMFT-equivalent projection

#### Result
The source bridge is in place, and the project has a clear understanding of why round-1 loses information.

---

### Stage_3 — Round-2 order-parameter refactor
Completed as the current baseline refactor stage.

#### Goal
Replace the restricted round-1 local pairing language with a more physical round-2 truth-layer channel language that still feeds the existing BTK workflow.

#### Completed items
- introduced `PhysicalPairingChannels`
- adopted the 8-channel round-2 language:
  - `delta_zz_s`
  - `delta_zz_d`
  - `delta_xx_s`
  - `delta_xx_d`
  - `delta_zx_s`
  - `delta_zx_d`
  - `delta_perp_z`
  - `delta_perp_x`
- formalized the split:
  - 7 core channels
  - 1 optional weak channel (`delta_zx_s`)
- updated `pairing.py` so the round-2 channels build the final `Delta(k)` used by BTK
- preserved compatibility with legacy `PairingParams`
- replaced hand-picked source projection with full-tensor round-2 fitting
- upgraded the fit to weighted ridge + global gauge fix
- unified projection metrics
- built a formal round-2 baseline from the low-temperature charge-balanced Luo cluster
- verified that round-2 channels can run through the current BTK forward pipeline

#### Result
The repository now has a usable round-2 pairing truth layer integrated into the physics pipeline.

#### Limitation inherited from Stage_3
Round-2 is clearly better than round-1, but the improvement is still modest. That is why the next work starts with residual anatomy before any further projection, baseline, or cleanup changes.
