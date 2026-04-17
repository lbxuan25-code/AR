# Projection Consistency Round 1

## 1. 背景

本报告只检查当前 `src/source/luo_projection.py` 的 round-1 投影实现是否自洽，
不修改 physics core，也不修改 projection 公式。

## 2. 当前 Projection 公式

- `eta_z_s = 0.5 * (delta_x[0,0] + delta_y[0,0]) * 1000`
- `eta_z_perp = 0.5 * (delta_z[0,2] + delta_z[2,0]) * 1000`
- `eta_x_s = 0.5 * (delta_x[1,1] + delta_y[1,1]) * 1000`
- `eta_x_d = 0.5 * (delta_x[1,1] - delta_y[1,1]) * 1000`
- `eta_zx_d = 0`, `eta_x_perp = 0`

## 3. Source 语义与单位检查

- `pairing_component_semantics`: `strong_evidence_support`. Loader names the six RMFT tensor slices as (chi_x, chi_y, chi_z, delta_x, delta_y, delta_z).
- `orbital_basis_alignment`: `strong_evidence_support`. H4band.py documents the source order as (dz2, dx2, dz2, dx2), which aligns with local (Az, Ax, Bz, Bx).
- `figure_level_channel_usage`: `strong_evidence_support`. plot_Fig1.py uses Pms[3], Pms[4], Pms[5] as delta_x, delta_y, delta_z and reads [1,1], [0,0], [0,2], matching the retained observables used by the local projection.
- `mu_unit_sanity`: `strong_evidence_support`. Mu2.npy[0] * 1000 nearly matches local BASE_MU_DIAG, strongly supporting the eV -> meV conversion for chemical potentials.
- `delta_unit_sanity`: `weak_evidence_support`. The same source npz payload family stores Mu fields in eV-scale numbers, and converting delta entries by 1000 moves them into the meV scale used by the local pairing baseline. This is supportive but not a direct proof from source-side comments.

结论口径：
- `Pms/pms` 六分量语义、轨道索引对应、以及 `Mu2.npy` 的 eV -> meV 量纲转换都有直接代码证据。
- `delta_*` 的 eV -> meV 对应目前属于弱证据支持，不应表述成已经被 source 注释严格证明。

## 4. 保留通道的一致性检查

- `eta_z_s`: max abs residual = 0.000e+00, mean abs residual = 0.000e+00, p95 abs residual = 0.000e+00
- `eta_z_perp`: max abs residual = 0.000e+00, mean abs residual = 0.000e+00, p95 abs residual = 0.000e+00
- `eta_x_s`: max abs residual = 0.000e+00, mean abs residual = 0.000e+00, p95 abs residual = 0.000e+00
- `eta_x_d`: max abs residual = 0.000e+00, mean abs residual = 0.000e+00, p95 abs residual = 0.000e+00

这部分是严格代数检查。若 residual 不接近 machine precision，应视为实现 bug。

## 5. Reconstruction Residual

- `R_x_xx_A`: max abs residual = 5.72858e-14 meV, mean abs residual = 1.55658e-15 meV, p95 abs residual = 7.10543e-15 meV
- `R_x_xx_B`: max abs residual = 4.13595 meV, mean abs residual = 0.110578 meV, p95 abs residual = 0.0547825 meV
- `R_y_xx_A`: max abs residual = 5.68712e-14 meV, mean abs residual = 1.57823e-15 meV, p95 abs residual = 7.10543e-15 meV
- `R_y_xx_B`: max abs residual = 4.147 meV, mean abs residual = 0.110637 meV, p95 abs residual = 0.0556117 meV
- `R_x_zz_A`: max abs residual = 69.9804 meV, mean abs residual = 6.08426 meV, p95 abs residual = 40.744 meV
- `R_x_zz_B`: max abs residual = 69.9804 meV, mean abs residual = 6.08448 meV, p95 abs residual = 40.744 meV
- `R_y_zz_A`: max abs residual = 69.9804 meV, mean abs residual = 6.08426 meV, p95 abs residual = 40.744 meV
- `R_y_zz_B`: max abs residual = 69.9804 meV, mean abs residual = 6.08442 meV, p95 abs residual = 40.744 meV
- `R_z_perp_02`: max abs residual = 2.84772e-14 meV, mean abs residual = 9.67334e-16 meV, p95 abs residual = 7.10543e-15 meV
- `R_z_perp_20`: max abs residual = 2.84217e-14 meV, mean abs residual = 9.80728e-16 meV, p95 abs residual = 7.10543e-15 meV

这部分说明：即使 projection 实现正确，本地参数化在 source-level retained observables 之外仍可能留下未建模残差。

## 6. 被忽略信息的规模

- `z-sector d-like omitted = 0.5*(delta_x[0,0]-delta_y[0,0])`: p95 abs = 40.744 meV
- `x_perp candidate` from symmetrized `delta_z[1,3]`: p95 abs = 17.1484 meV
- `zx_d candidate aggregate`: p95 abs = 9.69832 meV
- total retained ratio (`1 - residual/source`): p05 = 0.2669, median = 0.3467, p95 = 0.7340

更合理的 `zx_d` source analogue 候选条目：
- `delta_y_sym_01`: mean abs = 3.17839 meV, p95 abs = 10.2141 meV
- `delta_y_sym_23`: mean abs = 3.17838 meV, p95 abs = 10.2141 meV
- `delta_x_sym_01`: mean abs = 2.992 meV, p95 abs = 10.2143 meV
- `delta_x_sym_23`: mean abs = 2.99198 meV, p95 abs = 10.2143 meV

说明：这些条目来自 `delta_x/delta_y` 中 z-x mixed 的同层位置 `[0,1]`, `[2,3]`，
在结构上比 `delta_z` 的 interlayer mixed entries 更接近本地 `eta_zx_d` 的角色。

候选 `zx_d` source analogue 扫描结果：
- `delta_z_sym_03`: mean abs = 4.9647 meV, p95 abs = 31.8643 meV
- `delta_z_sym_12`: mean abs = 4.9647 meV, p95 abs = 31.8643 meV
- `delta_y_sym_01`: mean abs = 3.17839 meV, p95 abs = 10.2141 meV
- `delta_y_sym_23`: mean abs = 3.17838 meV, p95 abs = 10.2141 meV
- `delta_x_sym_01`: mean abs = 2.992 meV, p95 abs = 10.2143 meV
- `delta_x_sym_23`: mean abs = 2.99198 meV, p95 abs = 10.2143 meV

这里需要区分：
- 上述 omitted-channel 数值是严格统计结果。
- `zx_d` 的 source analogue 只是在 source tensor 上的候选 entry 诊断，不是最终严格物理定义。
- 全扫描里数值更大的 `delta_z_sym_03` / `delta_z_sym_12` 更像 interlayer mixed structure，不应直接等同于本地 `eta_zx_d`。

## 7. 最终判断

- “projection 实现正确”: Yes. The current luo_projection.py implementation matches its design formulas, and retained-channel residuals are at machine-precision scale.
- “projection 与 Luo 图口径一致”: Yes. The retained combinations align with the Luo figure-level s/d/perp observables used in plot_Fig1.py.
- “projection 与原 RMFT 全量信息完全一致”: No. Current round-1 projection is a restricted approximation to the RMFT source, not a full equivalence to the complete RMFT pairing tensor.

必须明确：当前 round-1 projection 是对 RMFT source 的受限近似映射，不应表述成与 full RMFT tensor 完全等价。

## 8. 下一步建议

- 如果后续发现 omitted channels 或 retained-ratio 统计持续偏大，应重新审视 `eta_zx_d`、`eta_x_perp` 和 z-sector anisotropy 的投影定义。
- 但本次任务只做诊断，不修改 `luo_projection.py`。