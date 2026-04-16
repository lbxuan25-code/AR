# Luo Source Map Round 1

## Source

- Repository: `https://github.com/ZhihuiLuo/RMFT_Ni327`
- Local inspected path: `/home/liubx25/Ni_Research/Projects/RMFT-guided surrogate for pairing-and-transport AR spectra/outputs/source/cache/RMFT_Ni327`
- Round-1 bridge assumption: Luo source energies are stored in eV and are converted to meV before mapping into local `PairingParams`.

## Files

| path | file_type | semantics | fields |
| --- | --- | --- | --- |
| Fig1.pdf | .pdf | paper figure output | - |
| Fig2.pdf | .pdf | paper figure output | - |
| Fig4.pdf | .pdf | paper figure output | - |
| Fig5.pdf | .pdf | paper figure output | - |
| Fig6.pdf | .pdf | paper figure output | - |
| H4band.py | .py | source-side plotting or model helper | - |
| Mu2.npy | .npy | auxiliary chemical-potential baseline table | array |
| README.md | .md | unclassified source artifact | - |
| func.py | .py | source-side plotting or model helper | - |
| pams_J0.09_0.18_0_0_0_0_complex_Nm220_-0.3372_0.3_-0.1646_0.3_121_93.npz | .npz | two-parameter px/pz RMFT phase-space scan | Js, Mu, N, Nm, Pms, eps, pxr, pzr |
| pams_fig1_alpha0.671965_J0.09_0.18_0_0_0_0_complex_Nm220_-0.501831_0.6_192.npz | .npz | doping sweep RMFT pairing data | Js, Mu, N, Nm, Pms, alpha, eps, pr |
| pams_figJ_p0_0_complex_Nm220_0_2_96_J3-0.03_JH0.npz | .npz | exchange-coupling sweep RMFT pairing data | JH, Jr, Mu, N, Nm, Pms, eps, p |
| pams_figJ_p0_0_complex_Nm220_0_2_96_JH-1.npz | .npz | exchange-coupling sweep RMFT pairing data | JH, Jr, Mu, N, Nm, Pms, eps, p |
| pams_figJ_p0_0_complex_Nm220_0_2_96_JH0.npz | .npz | exchange-coupling sweep RMFT pairing data | JH, Jr, Mu, N, Nm, Pms, alpha, eps, p |
| pams_figT_J0.09_0.18_0_0_0_0_p0_0_complex_T0_0.012_96.npz | .npz | temperature sweep RMFT pairing data | Js, Mu, N, Pms, Tr, eps, p |
| plot_Fig1.py | .py | source-side plotting or model helper | - |
| plot_Fig2.py | .py | source-side plotting or model helper | - |
| plot_Fig3.py | .py | source-side plotting or model helper | - |
| plot_Fig4.py | .py | source-side plotting or model helper | - |
| plot_Fig5.py | .py | source-side plotting or model helper | - |
| plot_Fig6.py | .py | source-side plotting or model helper | - |
| pms_p-0.15_-0.2_J0.09_0.18_0.npz | .npz | single RMFT sample snapshot | Js, mu, n, p, pms |
| pms_p-0.15_0_J0.09_0.18_0.npz | .npz | single RMFT sample snapshot | Js, mu, n, p, pms |
| pms_p0_-0.28_J0.09_0.18_0.npz | .npz | single RMFT sample snapshot | Js, mu, n, p, pms |
| pms_p0_0_J0.09_0.18_0.npz | .npz | single RMFT sample snapshot | Js, mu, n, p, pms |

## Usable Fields

- `Pms` / `pms`: RMFT pairing observables with six channel slices interpreted as `(chi_x, chi_y, chi_z, delta_x, delta_y, delta_z)`.
- `Mu` / `mu`: source chemical potentials. `Mu2.npy[0] * 1000` matches the migrated baseline `mu_diag`, so these arrays are treated as eV-scale source fields.
- `N` / `n`: orbital occupations or densities.
- sweep coordinates such as `pr`, `Tr`, `pxr`, `pzr`, `Jr`, and file-level metadata such as `Js`, `alpha`, `JH`, `eps`, `Nm`.

## Not Directly Usable Fields

- No direct local analogue was identified for the round-1 `eta_zx_d` channel.
- No direct local analogue was identified for the round-1 `eta_x_perp` channel.
- PDF figures are documentation artifacts rather than machine-readable source data.

## Projection Assumptions

| field | mode | source_expression | note |
| --- | --- | --- | --- |
| eta_z_s | approximate_inference | 0.5 * (delta_x[0,0] + delta_y[0,0]) * 1000 | Luo source resolves z-like diagonal x/y bond pairing; mapped to local z-like s component in meV. |
| eta_z_perp | direct_read | 0.5 * (delta_z[0,2] + delta_z[2,0]) * 1000 | Directly uses the interlayer z-like pairing matrix element and converts eV to meV. |
| eta_x_s | approximate_inference | 0.5 * (delta_x[1,1] + delta_y[1,1]) * 1000 | Maps Luo x-like diagonal x/y bond pairing to the local x-like s component in meV. |
| eta_x_d | approximate_inference | 0.5 * (delta_x[1,1] - delta_y[1,1]) * 1000 | Maps Luo x-like bond anisotropy to the local x-like d component in meV. |
| eta_zx_d | conventionally_zeroed | 0 | No direct RMFT field for the local z-x off-diagonal d channel was identified in round 1. |
| eta_x_perp | conventionally_zeroed | 0 | No direct RMFT field for the local x-like interlayer channel was identified in round 1. |

## Example Projected Sample

- Sample id: `pams_J0.09_0.18_0_0_0_0_complex_Nm220_-0.3372_0.3_-0.1646_0.3_121_93::0_0`
- Source file: `pams_J0.09_0.18_0_0_0_0_complex_Nm220_-0.3372_0.3_-0.1646_0.3_121_93.npz`
- Coordinates: `{'pxr': -0.3372, 'pzr': -0.1646}`
- Projected pairing params: `PairingParams(eta_z_s=(-0.004683436039405952+0.0004096163107733232j), eta_z_perp=(80.82441370816456-8.151260711162669j), eta_x_s=(0.6484646189770444+0.4054625723559333j), eta_x_d=(338.46416969778204-13.375317729445293j), eta_zx_d=0j, eta_x_perp=0j)`