# Pre-registered protocol: TRANS-002 — ST-GCN-scale taxonomy-transition cost replication on NTU60

> **Protocol ID**: PSD-NTU-TRANS-002 | **Status**: **FROZEN v1.0 2026-09-07**（用户令"算力不是问题，先补实验"；冻结先于任何运行）
> **Origin**: 恶意审稿 A3/A5——TRANS-001 的 MLP 规模使成本端点在算术上不可能通过（耦合臂 35s 无成本可省），且拒绝 matched-solver 对照。本协议在 ST-GCN 规模重跑双臂并加第三臂消除 solver 混杂。

## 1. Data / scenario (frozen — 与 TRANS-001 逐项同源)

`data/pyskl/ntu60_hrnet.pkl`（xsub 40,091 train / 16,487 val；17-joint HRNet 2D）；同一 Y→Y′ 十合并映射（60→49 类，`build_y_to_yp_map_ntu()` 原映射表，不重设计）。

## 2. Arms (frozen — 三臂)

- **Arm C′（coupled, ST-GCN 端到端）**：ST-GCN（与 E6/等价性验证同族配置：`psd/models` ST-GCN 300ep 惯例预算改为 80ep 与 D 臂对齐——**见 §6 披露 1**）端到端有监督从零训练于 Y′ 标签。
- **Arm D′（decoupled, ST-GCN）**：ST-GCN pretext 在 Y 时代自监督（AimCLR 目标，300ep，Y 全量标签不接触——与 NTU 等价性验证的 joint pretext 同源管线）训练一次后冻结，线性头在 Y′ 标签上重训。
- **Arm M（matched-solver 对照，新）**：**与 D′ 同一冻结特征 + 同一 solver 族**的头重训（StandardScaler + logistic regression，tol 1e-3，与 E9 系头一致），但分类器输入为 **C′ 端到端训练出的 ST-GCN 的 penultimate 特征**——即"耦合训练表示 + 解耦式重训成本"。M 的存在把"表示质量差"与"重训成本差"解耦：D′ vs M = 冻结自监督特征 vs 端到端有监督表示在同 solver 下的精度对比（成本同为线性头级）。

- **预算两档**：full（40,091）与 10%（4,009，seed 42 固定，与全系列同惯例）。seeds 42/43/44（成本比主终点；精度带同 E6 惯例）。

## 3. Decision rule (frozen)

- **CONFIRMS**: median wall-clock ratio C′/D′ ≥ 3× **且** |acc_D′ − acc_C′| < 2.3pp（E6 噪声带惯例）
- **PARTIAL**: ratio ≥ 3× 但精度差出带——如实报比值+代价
- **FAILS**: ratio < 3×——如实报；此时 **≥3× 声明在论文中的表述自动升级**：从"synthetic tier + MLP 规模下无成本可省"改为"synthetic tier + ST-GCN 规模复制仍 FAILS"（更强否定，L12 相应改写）
- M 臂不进决策规则，仅作机制对照（防事后挑选）：若 acc_M > acc_D′ ≥1pp，则"端到端有监督表示优于冻结自监督表示"成立，D 臂精度劣势部分归因表示而非范式。

## 4. Wall-clock accounting (frozen — 堵 TRANS-001 的"pretext 摊销"争议)

- D′ 的 pretext 训练（300ep，预计 ~4-6h）**单列**，计入 D′ 总成本一次，并同时报告摊销口径（ pretext ÷ 转换次数 = 1 次时的全摊与 N→∞ 的零摊两极）。
- C′ 端到端为每 seed 每档全量计时。
- 判定用 D′-head-only 口径（与 E6/TRANS-001 可比）；**同时报告** D′-amortized 口径——两个口径的 verdict 分列，若二者不一致，论文必须双报（防"挑有利口径"指控）。

## 5. Seeds / cost

Seeds 42/43/44。预算：pretext ~4-6h 一次性 + C′ full 3×~1.5h + C′ 10% 3×~10min + D′/M 头分钟级 ≈ **总计 ~9-12h GPU**。

## 6. Disclosures (frozen)

1. **epoch 对齐**：D′ pretext 300ep（自监督惯例），C′ 端到端 80ep——两者 epoch 数不同是有监督/自监督各自收敛惯例，非调参结果；80ep 选自 TRANS-001 冻结值以保持可比。C′ 的 300ep 敏感性检查（单 seed）作为附录诊断一并跑，防止"C′ 欠训练"指控。
2. 本协议冻结于见到任何 TRANS-002 结果之前；TRANS-001 结果已可见，这正是重跑动机，但**本协议的判读规则不依赖 TRANS-001 的任何数值**。
3. HRNet 2D 关节（非 3D Kinect）——与 E9b/TRANS-001 同源，如实沿袭。
4. M 臂特征提取自 C′ 的 80ep 检查点 penultimate。

## 7. Evidence

Driver: `scripts/run_r23_trans002.py`（待写）；evidence: `reports/r23-trans002-<date>.json`。

## 修订历史

| 版本 | 日期 | 说明 |
|---|---|---|
| v1.0 | 2026-09-07 | FROZEN：三臂设计（C′/D′/M matched-solver）+ 双口径 wall-clock 记账 + C′ epoch 敏感性检查。 |

## 2.1 Implementation details (dated 2026-09-07, frozen with v1.0 — 实现细节补充, 判读规则不变)

- **C′ 训练配方**（有监督 ST-GCN 社区惯例，非调参）：官方 AimCLR 代码链（`net.st_gcn.Model` + `Feeder_single`，frame-50 导出），SGD momentum 0.9，**batch 16 / lr 0.1 / weight_decay 1e-4 / step [60]**（ST-GCN 原论文惯例），80ep，官方 train 增强（shear/padding 按 pretext 配置）。
- **Y′ 标签进 Feeder**：label pkl 经冻结合并表映射为 49 类后另存 `*_yp49.pkl`（Feeder 零改动）。
- **D′ 头 = StandardScaler + logistic regression**（E9b/c 头惯例，tol 1e-3）：冻结 `epoch300_model.pt` 的 encoder penultimate(256) dump 后重训——与 M 臂同 solver 同头，唯 encoder 不同。
- **10% 档实现**：train 子集切片另存（4,009 clips, seed 42 分层），val 全量。
- **特征提取**：encoder penultimate 经 forward hook 获取；评估前处理与 linear-eval 逐键一致。
- **wall-clock**：外层 perf_counter 包络训练段 + GPU 快照前后各一次（TRANS-001 同口径）；bibtex 级噪声不滤波。

## Dated addendum 2026-09-09: PSD-NTU-TRANS-002-MEXP（扩容修订，先于任何新运行冻结）

范围：仅扩臂。冻结的三 seed 判据输出（seeds 42--44，PARTIAL，15.8x/−4.49pp，见上方 decision rule）**不变、不替换**；本修订在任何新训练开始前预先登记一项扩容：coupled 臂（C′）与其派生 matched-solver 臂（M）扩至十 seed（42--51；NaN 替换池 52--55，沿用 NaN 看门狗规则），把三 seed 的 M-vs-C′ 观察（冻结特征+线性头 82.25--82.83% vs coupled 全量微调 79.8%）升级为配对显著性检验。

预先声明的分析方案（先于任何新训练冻结）：
- 分析单元：逐 seed 配对。C′_s = 该 seed 的 max-accuracy checkpoint 在 49 类 Y′ 验证集上的净评估 top-1（eval 模式）；M_s = 在同一 checkpoint 的 penultimate 特征上用 decoupled solver（逻辑回归头）重训后的验证 top-1。
- 检验：十对 (M_s − C′_s) 的双侧 Wilcoxon signed-rank，α=0.05；同时报告 sign test、逐对数值、均值/中位数。
- 报告：无论显著与否均如实写入论文 TRANS-002 小节，标注为 post-hoc robustness expansion（dated，引用本修订）；冻结判据段落的措辞不改动。
- wall-clock：逐 seed 记录入工件，但不进入冻结成本比 verdict。
- 执行与恢复：沿用 fixfull 全套机制（save_interval=5、NaN 看门狗、断点续训、SYSTEM 看门狗任务、单实例锁）；无新鲜工件不得下任何结论。
