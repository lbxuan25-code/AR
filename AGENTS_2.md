# order_parameter_refactor_spec.md

## 任务定位

本文件定义 **AR 新仓库第二阶段：序参量层重构** 的正式规范。  
目标不是重写整个 physics core，也不是修改当前 KT/BTK transport solver，而是：

1. 保留当前正常态与 transport 主线；
2. 将当前 round-1 的受限 `PairingParams` / `delta_matrix(k)` 表示，升级为一个 **更高保真、具有明确物理含义** 的序参量层；
3. 让 Luo RMFT source 到本地序参量的映射，尽量从“手工挑几个矩阵元”升级为“物理通道层 + 受约束重构”；
4. 保持和当前 `SpectroscopyPipeline.interface_gap_diagnostics(...) -> compute_multichannel_btk_conductance(...)` 兼容。

---

## 一、必须先接受的当前事实

### 1. 当前 round-1 projection 不是 full RMFT 等价投影
当前投影一致性检查已经表明：

- 当前 `luo_projection.py` 对保留通道 `eta_z_s / eta_z_perp / eta_x_s / eta_x_d` 的实现是正确的；
- 但 current projection 只是 **restricted approximation**，不是 full RMFT pairing tensor 的完全等价投影；
- omitted 信息并不小，包括：
  - z-sector d-like
  - x-perp
  - zx-like mixed channels

因此，下一步不能再把 round-1 pairing 表示当作最终物理语言。

### 2. 当前 KT/BTK 求解器不需要被重写
当前 transport 流程已经建立在：

- `SpectroscopyPipeline.interface_gap_diagnostics(...)`
- `SpectroscopyPipeline.compute_multichannel_btk_conductance(...)`

之上。

**本次任务默认：不改 transport solver 核心原理。**  
需要改的是：
- pairing 参数容器
- pairing matrix 构造器
- Luo source -> local pairing representation 的映射层

### 3. 当前问题不是 projection 代码 bug，而是 local pairing ansatz 太小
也就是说：
- 不是“公式写错了”
- 而是“当前本地序参量语言截断过强，信息损失较大”

---

## 二、本次重构的核心原则

### 原则 A：不要把“低维拟合方便”放在“物理含义明确”前面
新的序参量层首先必须是：
- 与 Luo RMFT source 有自然对应关系
- 每个参数都有明确物理语义
- 可以解释为某个轨道扇区、某个 bond 对称性、或某类层间通道

### 原则 B：区分“物理通道层”和“BTK 接口层”
本次必须引入一个新的概念区分：

1. **物理通道层（physical channel layer）**
   - 这是仓库中正式的序参量语言
   - 参数具有明确物理含义
   - 直接对应 RMFT source 的轨道/bond/对称性通道

2. **BTK 接口层（BTK interface representation）**
   - 从物理通道层重建得到 `Delta(k)` / `delta_matrix(k)`
   - 供现有 projection / interface / KT(BTK) 流程使用

### 原则 C：重构不等于随意重组矩阵元
只有两类变换允许视为“物理语义保持”：

1. **可逆或近可逆的对称性重参数化**
2. **在 full source tensor 上误差可量化的受约束近似**

禁止：
- 无来源地手工拼参数
- 没有误差度量的拍脑袋简化

### 原则 D：本次优先重构 pairing，不动 normal-state
本轮不处理：
- normal-state 参数化升级
- TB 参数拟合
- transport kernel 物理公式替换

---

## 三、正式目标：建立新的“物理通道层”

请不要再把当前 round-1 的 `eta_z_s, eta_z_perp, eta_x_s, eta_x_d, eta_zx_d, eta_x_perp` 视为最终物理语言。

新的正式序参量层应至少覆盖以下通道：

### 1. z 扇区（dz2-like sector）
- `Delta_zz_s`
- `Delta_zz_d`

含义：
- `zz` 表示 z-like 轨道扇区
- `s/d` 表示来自 x/y bond pairing 的和差分解

### 2. x 扇区（dx2-y2-like sector）
- `Delta_xx_s`
- `Delta_xx_d`

### 3. z-x 混合扇区（mixed sector）
至少考虑：
- `Delta_zx_d`

如果实现自然，也允许进一步预留：
- `Delta_zx_s`

但 round-2 最小版本中，`Delta_zx_d` 优先级更高。

### 4. 层间通道（interlayer sector）
- `Delta_perp_z`
- `Delta_perp_x`

---

## 四、为什么选这组通道

这不是纯数学形式，而是源于你现在已经看到的 source 结构和 omitted diagnostics：

1. 当前 round-1 已保留：
   - `z_s`
   - `x_s`
   - `x_d`
   - `z_perp`

2. diagnostics 已明确显示当前还缺少至少三类重要信息：
   - z-sector d-like anisotropy
   - x-like interlayer pairing
   - z-x mixed pairing

因此 round-2 的最小高保真版本至少应补上：
- `z_d`
- `x_perp`
- `zx_d`

---

## 五、formal pairings 的定义方式

新的物理通道层应以 Luo source-native 的 bond pairing 为起点进行组织。

设 source 层已有：

- `delta_x`
- `delta_y`
- `delta_z`

且轨道基底顺序为 `(z, x, z, x)`。

则建议定义：

### z 扇区
\[
\Delta_{zz}^{s} = \frac{\delta_x^{zz} + \delta_y^{zz}}{2}
\]
\[
\Delta_{zz}^{d} = \frac{\delta_x^{zz} - \delta_y^{zz}}{2}
\]

### x 扇区
\[
\Delta_{xx}^{s} = \frac{\delta_x^{xx} + \delta_y^{xx}}{2}
\]
\[
\Delta_{xx}^{d} = \frac{\delta_x^{xx} - \delta_y^{xx}}{2}
\]

### mixed 扇区
优先定义一个最小 d-like mixed channel：
\[
\Delta_{zx}^{d}
\]
其精确定义不应拍脑袋写死，而应结合 source tensor 中候选 z-x mixed entries，通过诊断和约束选定。

### interlayer 扇区
\[
\Delta_{\perp z}
\]
\[
\Delta_{\perp x}
\]

---

## 六、重构后给 BTK 的接口形式

transport 仍然最终只吃 `Delta(k)`。

因此必须新增一个从“物理通道层”到 `Delta(k)` 的构造器。

建议形式：

### diagonal z sector
\[
\Delta_{zz}(k) = \Delta_{zz}^{s}\,\gamma_s(k) + \Delta_{zz}^{d}\,\gamma_d(k)
\]

### diagonal x sector
\[
\Delta_{xx}(k) = \Delta_{xx}^{s}\,\gamma_s(k) + \Delta_{xx}^{d}\,\gamma_d(k)
\]

### mixed sector
最小版本可先采用
\[
\Delta_{zx}(k) = \Delta_{zx}^{d}\,\gamma_d(k)
\]

如后续确有需要，再扩展为
\[
\Delta_{zx}(k) = \Delta_{zx}^{s}\,\gamma_s(k) + \Delta_{zx}^{d}\,\gamma_d(k)
\]

### interlayer sector
\[
\Delta_{\perp z}(k) = \Delta_{\perp z}
\]
\[
\Delta_{\perp x}(k) = \Delta_{\perp x}
\]

然后组装新的 4x4 pairing matrix。

---

## 七、要明确：这不是“又简化一层”的意思

重构后的“物理通道层”不是为了替代 RMFT source，也不是为了把模型压成更少参数。

它的作用是：

- 作为 **source-native 与 BTK interface 之间的正式物理语言**
- 让每个参数都有可解释的轨道/对称性含义
- 允许以后对不同通道加物理先验、做拟合约束、做鲁棒性分析

如果只是在 round-2 里用这套通道重构 `Delta(k)` 给 BTK，那么它不是额外简化，而是：

> **高保真物理表示层**

真正的低维简化，如果要做，也应该在 surrogate / inverse 的后续阶段单独定义，而不是让 physics truth layer 直接变成过小的拟合 ansatz。

---

## 八、投影方式必须升级

当前 round-1 做法更像：
- 取 source tensor 的少数矩阵元
- 直接映射成 local params

round-2 不应继续只靠“挑几个矩阵元”。

必须新增一个 **受约束投影 / 重构模块**。

### 推荐思路
给定 source tensor：
- `delta_x`
- `delta_y`
- `delta_z`

和一组待求的物理通道参数 `theta`，构造：
- `delta_x_recon(theta)`
- `delta_y_recon(theta)`
- `delta_z_recon(theta)`

再定义目标函数，例如：
\[
L(\theta)
=
w_x \| \delta_x^{src} - \delta_x^{recon}(\theta)\|_F^2
+
w_y \| \delta_y^{src} - \delta_y^{recon}(\theta)\|_F^2
+
w_z \| \delta_z^{src} - \delta_z^{recon}(\theta)\|_F^2
\]

在对称性约束下解最优 `theta`。

### 这样做的优点
- 不再只盯住几个矩阵元
- 可以系统量化误差
- 可比较不同 channel set 的 retained ratio / residual norm
- 更容易判断“这组物理通道是否足够”

---

## 九、本次不允许的做法

### 不要做
- 不要直接删除旧的 round-1 pairing 结构，除非新结构已经通过测试
- 不要同时改 normal-state
- 不要在没有 residual / retained-ratio 评估的前提下随意新增通道
- 不要把 source tensor 全部直接塞进 BTK solver，除非你能清楚解释新 `Delta(k)` 的构造规则
- 不要把“通道重组”写成单纯的数值黑箱 PCA / 无物理含义降维

---

## 十、需要新增或修改的模块

请优先使用下面的模块布局。

### 1. 新的参数容器
新增或重构：
- `src/core/parameters.py`

要求：
- 保留旧的 `PairingParams` 兼容层（如有必要）
- 新增正式的 round-2 物理通道容器，例如：
  - `PhysicalPairingChannels`
  - 或更合适的命名

### 2. 新的 pairing 构造器
修改：
- `src/core/pairing.py`

要求：
- 支持从新的物理通道层构造 `delta_matrix(k)`
- 明确保留 current form factors：`gamma_s`, `gamma_d`
- 让 mixed / interlayer channels 都能进入最终矩阵

### 3. source -> channel projection
新增：
- `src/source/round2_projection.py`

要求：
- 输入 Luo source sample
- 输出新的物理通道层参数
- 同时输出 projection residual / retained metrics

### 4. 诊断模块
新增：
- `src/source/round2_projection_diagnostics.py`

要求：
- 比较 round-1 和 round-2
- 输出：
  - retained ratio
  - source reconstruction residual
  - omitted-channel reduction
  - 代表性样本对比

---

## 十一、需要新增的脚本

### 1. round-2 投影脚本
- `scripts/source/build_round2_projection.py`

功能：
- 对 Luo samples 执行新的 round-2 投影
- 保存 projected channel params
- 保存 diagnostics summary

### 2. round-2 vs round-1 比较脚本
- `scripts/source/compare_round1_round2_projection.py`

功能：
- 比较两者在 retained ratio、残差、各通道规模上的改进
- 输出图和 summary

### 3. round-2 baseline forward smoke test 脚本
- `scripts/core/run_round2_baseline_forward.py`

功能：
- 用新的通道层构造 baseline / 示例样本的 `Delta(k)`
- 跑完整 physics forward + KT/BTK
- 证明 transport 主线仍然可用

---

## 十二、测试要求

必须新增测试：

- `tests/source/test_round2_projection_smoke.py`
- `tests/core/test_round2_pairing_matrix.py`
- `tests/core/test_round2_forward_smoke.py`

至少覆盖：

1. 新物理通道层可以成功构造 `delta_matrix(k)`
2. 对一个 Luo sample 可以完成 round-2 投影
3. round-2 的 retained ratio 不应劣于 round-1
4. round-2 forward workflow 能跑通到 BTK
5. 新旧 pairing API 若需兼容，兼容层必须可用

---

## 十三、输出要求

### 文档
新增：
- `docs/order_parameter_refactor_round2.md`

文档中必须说明：
1. 新物理通道层的定义
2. 每个通道的物理意义
3. source-native 到 local channel 的映射方式
4. channel -> `Delta(k)` 的构造规则
5. 为什么它比 round-1 更高保真
6. 它和 BTK solver 的接口关系

### 数据输出
至少生成：
- `outputs/source/round2_projection_summary.json`
- `outputs/source/round2_projection_examples.csv`
- `outputs/source/round1_vs_round2_projection_comparison.json`

可选图：
- retained ratio 对比图
- omitted norm 对比图
- representative source/reconstruction heatmap

---

## 十四、验收标准

只有同时满足以下条件，本次重构才算成功：

1. 新的物理通道层已经定义清楚，且参数具有明确物理语义；
2. 新的 pairing 构造器可以输出 `Delta(k)` 并送入现有 KT/BTK 流程；
3. round-2 投影在 retained ratio 或 reconstruction residual 上明显优于 round-1；
4. 新 `Delta(k)` forward smoke workflow 能跑通；
5. 文档清楚解释了“这不是单纯数学重组，而是 source-native 物理通道层重构”。

---

## 十五、最后提醒

这次任务的目标不是“让参数更多”。

这次任务的真正目标是：

> **把本地序参量语言从 round-1 的受限近似，升级为一个具有明确轨道/对称性/层间物理含义、并且能高保真重建 RMFT pairing source 的正式物理通道层。**

如果为了实现方便而继续把一部分明显非小的通道硬置零，那么这次重构就不算成功。
