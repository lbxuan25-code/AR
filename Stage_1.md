# AGENTS.md（Greenfield 版：从零新建 surrogate 仓库）

## 项目定位

本项目不是在现有 `LNO327_AR_Phenomenology` 仓库里继续开发，而是要**从零新建一个独立仓库**，用于：

1. 承载当前 LNO327 AR/BTK 物理 forward workflow 的**可复用核心**；
2. 接入 Luo 的 RMFT 数据作为 source；
3. 构建基于当前物理流程生成标签的训练数据集；
4. 训练 **pairing raw params + transport params -> AR spectrum** 的 surrogate；
5. 用 surrogate 加速第一轮 inverse demo；
6. 输出候选参数簇，而不是单点真值。

这个新仓库的本质是：

> **physics-core + source bridge + dataset builder + surrogate trainer + surrogate-assisted inverse demo**

不是一个只放神经网络脚本的仓库。

---

## 一、项目总原则

### 1. 物理 forward workflow 必须是仓库的一等公民
本新仓库中必须先建立或迁移一条清晰的物理 forward 主线：

\[
\text{ModelParams}
\rightarrow
\text{FS / projection / interface diagnostics}
\rightarrow
\text{multichannel BTK spectrum}
\]

surrogate 只能作为这条链的近似器，不能替代它。

### 2. 第一轮只做最小闭环
第一轮只完成：

- 仓库初始化
- physics core 迁移/复现
- Luo source 检查与桥接
- pairing+transport 数据集构建
- surrogate 训练与评估
- surrogate-assisted inverse 最小演示

### 3. 第一轮不处理 full normal-state inverse
normal-state 参数第一轮固定为 baseline，不纳入 surrogate 主训练空间。

### 4. 输出必须是参数簇，不是唯一真值
inverse 阶段只能输出：
- top-K candidate family
- parameter cluster
- near-optimal solution set

不能表述成“真实唯一参数”。

---

## 二、仓库初始化（必须首先完成）

这是一个从零新建仓库任务，所以第一阶段必须先完成基本工程骨架。

## 需要创建的目录结构

```text
repo_root/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   ├── core/
│   ├── surrogate/
│   ├── source/
│   ├── data/
│   └── utils/
├── scripts/
│   ├── core/
│   ├── source/
│   ├── dataset/
│   ├── surrogate/
│   └── inverse/
├── tests/
│   ├── core/
│   ├── source/
│   ├── dataset/
│   ├── surrogate/
│   └── inverse/
├── docs/
└── outputs/
```

## 初始化要求

### `README.md`
必须至少说明：
- 项目目标
- 第一轮范围
- 物理 forward workflow 是真值
- surrogate 是近似器
- inverse 输出是参数簇

### `pyproject.toml`
必须设置：
- 项目名称
- Python 版本
- 核心依赖
- 开发依赖
- 测试配置（如可行）

### `.gitignore`
必须忽略：
- `__pycache__/`
- `.pytest_cache/`
- `.venv/`
- 大型数据缓存
- 模型权重输出
- 中间训练日志

---

## 三、依赖要求

第一轮建议依赖保持克制。

### 最少依赖
- `numpy`
- `scipy`
- `matplotlib`
- `pytest`
- `torch`
- `pydantic` 或 dataclass-based config（任选一种风格）
- 如需要可加 `pandas`

### 不要在第一轮引入
- 重型 MLOps 框架
- 大型分布式训练框架
- 复杂数据库

---

## 四、physics core 迁移/复现阶段

## 目标
在新仓库里先建立一个独立可运行的 physics forward core。

这一步必须发生在 surrogate 之前。

## 允许的实现方式（优先级从高到低）

### 方案 A（推荐）
从当前 `LNO327_AR_Phenomenology` 中**选择性迁移**最小必要核心模块到新仓库。

### 方案 B
把原仓库当成外部依赖，再在新仓库里封装调用层。

### 方案 C（不推荐）
从头手写一套新的 physics core。

第一轮优先使用方案 A，避免 physics 和 surrogate 同时重写。

---

## 需要落地的 physics core 模块

在 `src/core/` 下至少建立：

- `parameters.py`
- `normal_state.py`
- `pairing.py`
- `bdg.py`
- `projection.py`
- `fermi_surface.py`
- `interface_geometry.py`
- `interface_gap.py`
- `btk_multichannel.py`
- `pipeline.py`

## 最小职责

### `parameters.py`
定义：
- `NormalStateParams`
- `PairingParams`
- `ModelParams`

### `normal_state.py`
定义固定解析 normal-state Hamiltonian

### `pairing.py`
定义统一多通道 pairing matrix

### `projection.py`
实现 band-basis projection

### `interface_gap.py`
实现 `Delta_plus / Delta_minus` 诊断

### `btk_multichannel.py`
实现多通道 broadened conductance forward solver

### `pipeline.py`
将上述步骤封装成统一 forward workflow

---

## 五、第一轮 baseline 要求

在新仓库中必须建立一套 baseline 配置。

第一轮至少需要：

- `base_normal_state_params()`
- `base_pairing_params()`
- `base_model_params()`

### 重要说明
即便这个是新仓库，也不要把 baseline 定义成“任意占位符”。

baseline 可以来自：
- 现有仓库正式 baseline
- 或经明确记录的迁移版本

但必须可追踪。

同时在文档中明确：
- baseline 的来源
- 是否与旧仓库一致
- 哪些地方只是迁移复刻

---

## 六、必须先跑通的 physics smoke workflow

在 surrogate 之前，必须保证新仓库已经能从参数出发生成谱线。

## 需要新增的脚本

- `scripts/core/run_baseline_forward.py`

## 这个脚本必须完成
输入：
- baseline model params
- 一组固定 transport 参数

输出：
- 一张 baseline spectrum 图
- 一份 summary JSON
- 若可行，gap-on-FS 图

## 如果这个脚本跑不通
禁止开始任何 surrogate 开发。

---

## 七、Luo source 接入（source layer）

## 外部 source
用户指定的 Luo 仓库：
- `https://github.com/ZhihuiLuo/RMFT_Ni327`

## 第一阶段必须先做检查
在 `scripts/source/inspect_luo_source.py` 中：

1. 检查 Luo 仓库结构；
2. 找出 RMFT 数据文件；
3. 记录文件语义；
4. 识别可直接读取字段；
5. 识别需要投影/映射的字段；
6. 识别缺失字段。

## 必须输出的文档
- `docs/luo_source_map_round1.md`

该文档必须包含：
- 路径
- 文件类型
- 语义说明
- 可用字段
- 不可用字段
- 投影假设

---

## 八、source bridge 设计

在 `src/source/` 下建立：

- `luo_loader.py`
- `luo_projection.py`
- `schema.py`

## `schema.py`
定义内部标准 sample schema，例如：
- sample_id
- doping / label
- source metadata
- source pairing observables
- source chemical potential（若有）
- projected pairing params
- projection provenance

## `luo_loader.py`
负责读取 Luo source 文件

## `luo_projection.py`
负责将 Luo source 映射到当前仓库 `PairingParams`

### 必须显式记录
每个输出项属于：
- 直接读取
- 近似推断
- 约定置零
- 暂无法确定

---

## 九、第一轮 surrogate 的参数空间

第一轮只固定 normal state，不训练它。

## 固定部分
- `NormalStateParams = base_normal_state_params()`

## 训练变量
### Pairing raw params
- `eta_z_s`
- `eta_z_perp`
- `eta_x_s`
- `eta_x_d`
- `eta_zx_d`
- `eta_x_perp`

### Transport
- `barrier_z`
- `gamma`

可选固定：
- `interface_angle = 0.0`
- `temperature = formal baseline value`

---

## 十、gauge fixing

pairing raw params 为复数，存在全局相位冗余。

在 `src/surrogate/raw_space.py` 中必须实现：

- `pairing_params_to_gauge_fixed_vector(...)`
- `gauge_fixed_vector_to_pairing_params(...)`

### 默认规范
- 优先令 `eta_z_perp` 为实数且非负
- 若 `|eta_z_perp|` 太小，使用明确写入文档的 fallback gauge

### 要求
- forward 训练样本必须统一使用 gauge-fixed 表示
- round-trip 数值稳定
- 不能把全局相位冗余交给神经网络去学

---

## 十一、数据集构建层

在 `src/data/` 下建立：

- `dataset_builder.py`
- `splits.py`
- `manifest.py`

在 `scripts/dataset/` 下建立：

- `build_pairing_transport_dataset.py`

## 数据集来源
必须包含三类样本：

### A. Luo 投影样本
作为物理可信 source anchor

### B. Luo 样本附近局部扰动
增强 source manifold 附近的平滑泛化

### C. transport 扫描样本
扫描：
- `barrier_z`
- `gamma`

---

## 十二、第一轮数据集规模

第一轮不要做大而全。

### 推荐分级
- smoke：500 ~ 1000
- mini：2000 ~ 5000
- full round1：10000 左右

默认以 **mini -> full round1** 的节奏推进。

---

## 十三、标签定义

第一轮 surrogate 的主标签为：

- 固定 bias grid 上的完整 broadened conductance 向量

## bias grid 规则
统一固定，例如：
- `bias_max = 40 meV`
- `num_bias = 601`

所有训练样本必须使用同一 grid。

## 可选辅助特征
可附带保存：
- `zero_bias_conductance`
- `dynamic_range`
- peak positions
- edge flags

但主训练任务必须仍是整条谱线回归。

---

## 十四、surrogate 模型层

在 `src/surrogate/` 下建立：

- `models.py`
- `train.py`
- `evaluate.py`
- `inverse.py`
- `config.py`

在 `scripts/surrogate/` 下建立：

- `train_pairing_transport_surrogate.py`
- `evaluate_pairing_transport_surrogate.py`

## 第一版模型要求
优先使用：
- MLP
- 或 residual MLP

### 第一轮不允许
- transformer
- diffusion
- 复杂 flow-based generative model

---

## 十五、训练要求

### 输入
- gauge-fixed pairing raw parameter vector
- `barrier_z`
- `gamma`

### 输出
- 601 维 conductance 向量

### 训练必须支持
- train/val/test split
- 固定 random seed
- early stopping
- checkpoint 保存
- config 导出
- 训练日志保存

### loss 建议
至少包括：
1. 曲线 MSE
2. 一阶导数误差（推荐）
3. 对低偏压和峰区适度加权（推荐）

---

## 十六、评估要求

在 `scripts/surrogate/evaluate_pairing_transport_surrogate.py` 中必须至少输出：

1. test MSE
2. 一阶导数误差
3. 零偏电导误差
4. peak position 误差
5. dynamic range 误差
6. 最差样本图
7. 代表性样本图
8. Luo-source-like held-out 子集评估

### 特别要求
必须单独检查：
- 低偏压区域
- coherence peak 区域

如果 surrogate 只在高偏压平坦区表现好，但低能结构严重失真，则第一轮不能算完成。

---

## 十七、inverse 最小演示层

在 `scripts/inverse/` 下建立：

- `run_inverse_with_surrogate.py`

## 第一轮 inverse 只做 demo
流程如下：

1. 输入一条目标谱
   - 可先用模拟谱代替真实实验谱
2. surrogate 快速搜索参数空间
3. 取前若干候选
4. 用**真实 physics forward workflow**重新计算这些候选
5. 输出：
   - top-K candidates
   - 参数簇表格
   - 谱线对比图
   - 残差图

### 关键要求
- surrogate 不能直接决定最终答案
- 必须由真实 forward 对候选复核
- 输出的是候选簇，不是唯一真值

---

## 十八、normal-state 决策规则

第一轮默认：
- 不训练 normal-state 参数

只有当第一轮完成后，出现明显系统残差，且 pairing+transport 不足以解释时，第二轮才允许逐步加入：

1. `mu_diag`
2. `e1`, `e2`
3. `vx`, `v1`, `v2`, `vxz`

第一轮绝不加入全部 hopping 参数。

---

## 十九、测试要求

在 `tests/` 下新增：

- `tests/core/test_forward_smoke.py`
- `tests/source/test_luo_loader.py`
- `tests/source/test_luo_projection.py`
- `tests/surrogate/test_raw_space.py`
- `tests/dataset/test_dataset_smoke.py`
- `tests/surrogate/test_train_smoke.py`
- `tests/inverse/test_inverse_smoke.py`

### 至少覆盖
1. baseline forward 能跑通
2. Luo source 至少一条样本可读取
3. 至少一条样本能映射到 `PairingParams`
4. gauge fixing 有效
5. raw vector round-trip 正常
6. 小数据集构建 smoke test
7. 小模型训练 smoke test
8. inverse smoke test

---

## 二十、必须生成的文档与输出

## 文档
- `docs/luo_source_map_round1.md`
- `docs/greenfield_repo_design.md`
- `docs/surrogate_round1_design.md`

## 输出目录
- `outputs/core/...`
- `outputs/source/...`
- `outputs/dataset/...`
- `outputs/surrogate/train/...`
- `outputs/surrogate/eval/...`
- `outputs/inverse/...`

---

## 二十一、第一轮成功标准

只有同时满足以下条件，第一轮才算完成：

1. 新仓库已初始化并可安装运行；
2. physics forward baseline script 能跑通；
3. Luo source 检查与桥接完成；
4. pairing+transport 数据集可从头构建；
5. surrogate 能在 held-out 样本上预测整条谱；
6. surrogate 对低能结构没有完全失真；
7. surrogate-assisted inverse demo 可跑通；
8. inverse 输出的是候选参数簇而不是伪单点答案。

---

## 二十二、最终汇报要求

完成后必须汇报：

1. 新仓库目录结构
2. physics core 是如何迁移/复现的
3. baseline 的来源
4. Luo source 如何接入
5. `PairingParams` 的投影规则
6. gauge fixing 规则
7. 第一轮数据集规模
8. surrogate 模型结构
9. 训练/验证/测试误差
10. inverse top-K 候选结果
11. 是否建议第二轮加入 normal-state 参数

---

## 最终提醒

这个 greenfield 仓库的第一目标不是“尽快用 AI 拟实验”。

第一目标是：

> **先把 physics core、source bridge、dataset、surrogate、inverse demo 串成一个可复现、可测试、可扩展的最小闭环。**

任何跳过 physics core、直接写神经网络脚本的做法都不合格。
