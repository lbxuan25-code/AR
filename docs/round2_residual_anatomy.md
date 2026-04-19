# Round-2 Residual Anatomy

## Scope

This note only audits where the current round-2 truth-layer reconstruction still misses the Luo source.
It does not change BTK, surrogate, inverse, or normal-state logic.

## Main Finding

- Dominant cause: `missing_channel_structure`
- Diagnostic note: Most residual weight sits on entries outside the current round-2 supported masks, so the mismatch is more likely dominated by missing source-channel structure than by the current fit weights.
- Median-dominant residual block: `delta_z`

## Block Breakdown

- `delta_x`: median residual norm = 25.7242, median retained ratio = 0.2937
- `delta_y`: median residual norm = 27.1621, median retained ratio = 0.2940
- `delta_z`: median residual norm = 33.5463, median retained ratio = 0.4064

## Residual Hotspots

- `delta_x` top hotspot `[0, 2]` in group `other` with mean abs residual 19.034 meV
- `delta_y` top hotspot `[0, 2]` in group `other` with mean abs residual 20.2182 meV
- `delta_z` top hotspot `[2, 2]` in group `other` with mean abs residual 23.3784 meV

## Representative Samples

- `best`: `pams_fig1_alpha0.671965_J0.09_0.18_0_0_0_0_complex_Nm220_-0.501831_0.6_192::0`, retained ratio = 1.0000, dominant block = `delta_x`, hotspot = delta_x[1,1]
- `median`: `pams_J0.09_0.18_0_0_0_0_complex_Nm220_-0.3372_0.3_-0.1646_0.3_121_93::105_35`, retained ratio = 0.3663, dominant block = `delta_x`, hotspot = delta_z[0,0]
- `worst`: `pams_J0.09_0.18_0_0_0_0_complex_Nm220_-0.3372_0.3_-0.1646_0.3_121_93::107_88`, retained ratio = 0.2631, dominant block = `delta_y`, hotspot = delta_z[2,2]
