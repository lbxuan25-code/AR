# pairing_state_next_stage_plan.md

## 文档目的

本方案用于指导 `/AR` 仓库下一阶段对 **pairing state / 序参量层** 的继续修改。  
目标不是继续机械增加参数，而是把当前已经落下来的 round-2 物理通道层，推进到一个：

- **物理含义明确**
- **与 Luo RMFT source 有稳定对应**
- **可作为 BTK truth model 使用**
- **对后续 surrogate / inverse 拟合更鲁棒**

的正式版本。

---

## 一、当前仓库现状

### 1. 已经完成的升级

当前仓库已经从 round-1 的 `PairingParams` 升级出了新的 round-2 物理通道层 `PhysicalPairingChannels`，包含 8 个复通道：

- `delta_zz_s`
- `delta_zz_d`
- `delta_xx_s`
- `delta_xx_d`
- `delta_zx_s`
- `delta_zx_d`
- `delta_perp_z`
- `delta_perp_x`

这说明仓库已经不再停留在旧的 6 参数近似语言，而是开始采用“轨道扇区 + 对称性 + 层间结构”的物理通道组织方式。fileciteturn388file0

`src/core/pairing.py` 也已经支持从新的物理通道层构造最终的 `delta_matrix(k)`，并保持与旧 `PairingParams` 的兼容转换。fileciteturn389file0

### 2. source 投影也已经进入 full-tensor 拟合阶段

当前 `src/source/round2_projection.py` 已经不再只抓几个矩阵元，而是：

1. 读取 Luo source 中的 `delta_x`, `delta_y`, `delta_z`
2. 建立 round-2 的 basis tensors
3. 对 full source tensor 做复数最小二乘拟合
4. 得到 8 个 physical channels 以及重构指标

这一步比 round-1 更接近“高保真物理通道层”。fileciteturn390file0

### 3. 现有 BTK 流程已经能接入新的 pairing state

当前 `scripts/core/run_round2_baseline_forward.py` 与 `tests/core/test_round2_forward_smoke.py` 已经证明：

- 新的 `PhysicalPairingChannels`
- 可以组装进 `ModelParams`
- 可以进入 `SpectroscopyPipeline`
- 可以一路跑到 `compute_multichannel_btk_conductance(...)`

因此，“新序参量层能否带入现有 KT/BTK 计算流程”这个问题，当前答案已经是 **可以**。fileciteturn394file0 fileciteturn400file0

---

## 二、当前仍存在的关键问题

虽然框架已经落下来了，但目前还不能把 round-2 直接当成最终正式版本。

### 问题 1：通道虽然增加了，但还没有“正式定型”
当前 8 个通道是：

- `zz_s`
- `zz_d`
- `xx_s`
- `xx_d`
- `zx_s`
- `zx_d`
- `perp_z`
- `perp_x`

其中真正由前期 round-1 omitted-channel diagnostics 强烈支持必须补进来的，是：

- `zz_d`
- `perp_x`
- `zx_d`

而 `zx_s` 虽然已经加入，但其必要性和稳定性目前并不如前三者强。

### 问题 2：当前 round-2 投影还是“无约束最小二乘”
虽然代码与文档口径里把它称为 “constrained complex least-squares reconstruction”，但就当前实现看，实际使用的是：

- `np.linalg.lstsq(design, target, rcond=None)`

也就是 **无显式物理约束的最小二乘**。fileciteturn390file0

这意味着：

- 数值上它能找到最佳线性逼近
- 但物理上它不一定最稳
- 也不一定最适合作为以后 inverse 的正式参数层

### 问题 3：round-2 目前只是“略有改进”，还不是质变
当前 comparison summary 给出的结果是：

- round-1 median retained ratio total: `0.3467`
- round-2 median retained ratio total: `0.3663`
- median retained-ratio improvement: `0.0217`
- round-1 median residual norm total: `59.27`
- round-2 median residual norm total: `54.49`

说明 round-2 **确实优于 round-1**，但提升幅度还不够大。fileciteturn397file0

### 问题 4：没有真正的 round-2 baseline
当前 `base_physical_pairing_channels()` 只是把旧 baseline 兼容性地翻译到新通道层，并把新增通道初始化为 0。  
它本质上不是一个真正由 round-2 物理语言定义出来的正式基准态。fileciteturn395file0

### 问题 5：投影评价指标前后定义不统一
目前仓库里不同阶段的 retained ratio / omitted fraction / residual norm 口径存在不一致风险。  
如果不尽快统一，后续一旦进入 surrogate / inverse，就会造成解释混乱。

---

## 三、下一阶段总目标

下一阶段的目标不是“让参数更多”，而是：

> **把当前 round-2 pairing state 从“可运行的框架版本”，推进到“正式可用的 physics truth layer”。**

这意味着它应该满足：

1. 通道定义稳定；
2. 与 Luo source 的映射有物理约束；
3. 能作为 BTK truth model；
4. 和后续 surrogate / inverse 参数层可以分层衔接。

---

## 四、建议的修改路线

### 阶段 A：先把正式通道集定型

#### 建议
将当前 8 通道划分为：

### 核心正式通道（7 个）
- `delta_zz_s`
- `delta_zz_d`
- `delta_xx_s`
- `delta_xx_d`
- `delta_zx_d`
- `delta_perp_z`
- `delta_perp_x`

### 可选弱通道（1 个）
- `delta_zx_s`

#### 原因
从已有 omitted-channel diagnostics 看，最明显需要补进来的通道是：

- z-sector d-like
- x-like interlayer
- z-x mixed d-like

而 `zx_s` 虽然有可能存在，但目前没有同等级别的强证据表明它必须和其余通道同权。

#### 执行建议
- 暂时保留 `delta_zx_s` 的代码支持
- 但在 projection 和 baseline 方案中，将其视作 **弱通道 / 正则化优先压小的通道**
- 文档中把它明确标记为 optional mixed channel

#### 验收目标
- pairing state 物理结构更清晰
- 避免无必要的参数膨胀
- 为后续受约束投影创造更稳的通道层

---

### 阶段 B：把 source -> channels 的投影升级成“带物理约束的拟合”

这是下一阶段最关键的一步。

#### 当前状态
现在是：

\[
\theta = \arg\min \|A\theta-b\|_2^2
\]

即纯复数最小二乘。fileciteturn390file0

#### 下一步建议
改成：

\[
L(\theta)
=
\|A\theta-b\|_2^2
+
\lambda_{zx_s}|\Delta_{zx}^s|^2
+
\lambda_{zx_d}|\Delta_{zx}^d|^2
+
\lambda_{\perp x}|\Delta_{\perp x}|^2
+
L_{\text{phase/gauge}}
\]

即至少加入：

1. **channel regularization**
2. **phase / gauge control**
3. **不同 source tensor 分块的加权**

#### 具体建议

##### 1. 通道正则化
- 对证据较弱、波动可能较大的通道加更强正则
- 尤其是 `delta_zx_s`
- `delta_perp_x`、`delta_zx_d` 可加较弱正则
- `delta_zz_s`、`delta_xx_s`、`delta_xx_d`、`delta_perp_z` 作为主通道，正则应更弱

##### 2. gauge / phase 约束
至少定义：
- 全局相位规范
- 主要通道的相位参考
- 是否优先鼓励某些主通道相对相位接近 0 或 π

##### 3. source block 权重
不要让 `delta_x`、`delta_y`、`delta_z` 的所有矩阵元一视同仁。  
建议定义：

\[
L =
w_x \|\delta_x^{src}-\delta_x^{recon}\|_F^2
+
w_y \|\delta_y^{src}-\delta_y^{recon}\|_F^2
+
w_z \|\delta_z^{src}-\delta_z^{recon}\|_F^2
+
L_{reg}
\]

必要时还可以：
- 对 pocket-sensitive entries 加权
- 对明显噪声型 entries 降权

#### 为什么这一步最重要
因为你未来想要的是：
- 具有精确物理含义
- 且拟合鲁棒

如果仍然用无约束 `lstsq`，它数学上最优，但物理上不一定稳。

#### 验收目标
- retained ratio 比当前 round-2 再提升
- residual norm 再下降
- 通道分布更稳定
- optional channels 不再出现数值过拟合

---

### 阶段 C：建立真正的 round-2 baseline

#### 当前问题
现在的 `base_physical_pairing_channels()` 只是兼容旧 baseline，并不代表真实的 round-2 物理基准态。fileciteturn395file0

#### 建议方案

### 方案 A：单样本 baseline
从 Luo source 中选一个最接近你当前 baseline 物理语境的参考样本，例如：
- 低温
- 参考掺杂
- 最接近你当前 used baseline 的 sample

然后用新的 round-2 投影得到正式 baseline。

### 方案 B：参考样本簇 baseline（更推荐）
选择一簇物理上相近的 Luo samples：
- 相同或邻近掺杂
- 相同低温区
- 同类解支

然后对每个通道：
- 取中位数
- 或取带异常点剔除的平均值

形成：
- `base_round2_physical_pairing_channels()`

#### 我更推荐的理由
- 对单个 outlier 不敏感
- 更适合后续 surrogate / inverse
- 更像正式物理基准态，而不是单点巧合

#### 验收目标
- 仓库中存在真正的 round-2 baseline
- 不是简单的 round-1 兼容映射
- 后续所有 round-2 forward / dataset / surrogate 都以它为基准

---

### 阶段 D：统一 projection 指标定义

#### 当前问题
目前不同阶段里 retained ratio 的定义不完全一致，已经出现明显数值口径差异。

#### 建议
将以下指标的定义写成仓库内唯一标准函数：

- `retained_ratio_total`
- `retained_ratio_x/y/z`
- `residual_norm_total`
- `omitted_fraction_total`

并在：
- round-1 diagnostics
- round-2 diagnostics
- future round-3 diagnostics
- surrogate quality audit

里全部复用。

#### 验收目标
- 指标定义唯一
- comparison 文件之间可直接横向比较
- 文档不再出现“名字一样但定义不同”的情况

---

### 阶段 E：明确 truth pairing state 和 fit pairing state 的分层

这是后续做 surrogate / inverse 前必须定下来的结构。

#### 建议分层

### 1. truth pairing state
使用：
- round-2 physical channel layer
- 受约束 source projection
- 高保真 `Delta(k)` 重建

用途：
- 作为 BTK forward 的 physics truth layer
- 生成训练数据
- 做 source consistency checks

### 2. fit pairing state
以后为 surrogate / inverse 单独定义较低维参数层，例如：
- 主通道幅值
- 少量相对相位
- 个别弱通道的控制参数

用途：
- 参数扫描
- 实验拟合
- 不确定性分析

#### 原则
不要让“拟合方便”直接污染“真值层”。

#### 验收目标
- pairing truth layer 与 fit layer 概念分清
- 后续 surrogate 不再直接绑定所有 truth channels
- 拟合时更稳，物理解释也更清楚

---

## 五、推荐的执行顺序

### Step 1
先把通道正式分成：
- 核心 7 通道
- 可选弱通道 `delta_zx_s`

### Step 2
实现受约束 / 加正则 / 带权重的 round-2 projection

### Step 3
建立正式 `base_round2_physical_pairing_channels()`

### Step 4
统一 retained ratio / residual / omitted 指标定义

### Step 5
再评估：
- 当前 round-2 是否已经足够做 BTK truth model
- 是否还需要继续扩通道

### Step 6
最后再基于新的 truth pairing state，定义 future surrogate / inverse 用的低维 fit pairing state

---

## 六、本阶段不建议做的事情

### 不要做
- 不要继续机械增加更多通道
- 不要现在就大规模用当前 8 通道直接拟实验
- 不要把当前 round-2 视为终版 pairing state
- 不要在未统一指标前继续扩展 round-3 文档
- 不要同时修改 normal-state 和 pairing state

---

## 七、预期结果

如果按本方案推进，下一阶段结束后应能得到：

1. 一个正式定型的 round-2 pairing truth layer；
2. 一个带物理约束的 source -> channel 投影器；
3. 一个真正的 round-2 baseline；
4. 一套统一、可比较的 projection 指标；
5. 一个与 future surrogate / inverse 更好解耦的 pairing state 体系。

---

## 八、最终结论

下一阶段继续修改 pairing state，最重要的不是“再加参数”，而是：

> **把现有 round-2 物理通道层收紧并物理化：定型核心通道、把投影升级成带物理约束的拟合、建立真正的 round-2 baseline，并统一 projection 指标定义。**

只有这样，pairing state 才能真正从“可运行的重构框架”走向“正式可用的 physics truth layer”。
