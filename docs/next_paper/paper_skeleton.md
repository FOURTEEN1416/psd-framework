# Next Paper — Full Skeleton (v0.1, 2026-11-20)

> 工作题：**Anchor Recycling: Label-Efficient Skeleton Behavior Recognition Under
> Evolving Taxonomies**（标注双用 = 同一锚点预算既塑形物理层又驱动语义层）。
> 本文件是全文骨架：每节给出可直接扩写的论点、段落要点与实测数字（来源见
> `reports/p1_sprint_evidence/adaptation/`）。目标期刊：TMM / PRL（无版面费）。
> 状态标记：✅ 数字已实测入表；🟡 待补实验；✍️ 待扩写。

---

## Abstract（✍️ 骨架已定，✅ 数字齐）

Skeleton behavior recognition under evolving taxonomies faces a recurring
problem: when the label taxonomy changes or the domain shifts, a frozen
self-supervised physical layer caps downstream accuracy, and the standard
remedy — full fine-tuning — destroys the label efficiency that made the
framework attractive. We study a setting we call **weak-reference domains**,
where the annotation budget is small (5–15% of clips) and the reference is
noisy. Through a controlled four-tier study (NTU60 → PanAf500 → NTU120 →
UCF101) we first show that **every label-free route to adapting a frozen
encoder fails**: test-time entropy minimization, low-rank continued
pretraining, cross-domain transfer from a strong source, and pseudo-label
feedback all move retention by at most ±1 pp (four negatives, one ruler).
We then show that the same small anchor set can be **recycled**: using the
identical 10% anchors to both shape the encoder (low-rank supervised
fine-tuning with a distillation anchor) and drive the semantic layer yields
+10.2 pp on NTU120 (74.96% → 85.16%, PARTIAL), +9.7 pp on PanAf500
(82.5% → 92.21%, CONFIRMS), and +20.1 pp on UCF101 (51.09% → 71.21%) — with
the gain growing as reference quality degrades. The effect has a two-sided
envelope: too little anchor mass (40 anchors) leaves outcomes unchanged, and
too much adaptation pressure trips the semantic layer's calibration gate. We
map the anchor-mass/learning-rate envelope across tiers and release the
unified measurement harness.

**Keywords**: skeleton action recognition, domain adaptation, label
efficiency, self-supervised learning, evolving taxonomies.

---

## 1. Introduction（✍️ 待扩写；论点已定）

- 段 1：演化分类学场景（NTU60→NTU120 merge/rename、新域接入）中，标注预算永远稀缺；
  冻结物理层 + 语义层适配（PSD）在强参照域有效，但在弱参照域封顶（E9 四层梯度
  90.7/82.5/74.96/51.09）。
- 段 2：自然的问题是"让冻结编码器动起来"。我们系统检验四条零标注路线并全部证伪
  （Finding 1）——这是本文的**边界地图**贡献。
- 段 3：关键观察：失败的原因不是"编码器不能动"，而是"没有真标签塑形它"。由此提出
  **anchor recycling（标注双用）**：同一锚点集既驱动语义层又低秩微调物理层。
- 段 4：贡献列表：
  (1) 弱参照域设定与四层统一测量协议（同一把尺、预注册判据）；
  (2) 四条零标注适配路线的系统性证伪（含 BN 口径陷阱、校准门禁等诊断）；
  (3) 标注双用机制：三域 +10.2/+9.7/+20.1pp，两域过预注册判定线；
  (4) 适配强度双侧包络（锚点质量下限 + 校准门禁上限）与剂量响应映射。

## 2. Related Work（✍️ 待扩写）

- 自监督骨架预训练：AimCLR、CrosSCLR、BehaviorMAE、TREBA/ProSL（对应 R3 意见）；
- 域适配/测试时适配：TENT、源自由 DA；持续预训练与灾难性遗忘；
- 类别增量与语义演化：LwF、merge/rename 零重训界（E1）；
- 标签高效学习：半监督、伪标签自训练、SupCon；
- 与本文差异：本文不提新 SSL 目标，而是**系统测量"编码器该不该动、怎么动"**，
  并给出双用机制与双侧包络。

## 3. Method（✍️ 骨架已定）

### 3.1 PSD 回顾
冻结物理层（AimCLR pretext）+ 语义层（规则种子 → 锚点聚类 → 置信过滤伪标签 →
自训练头）。E9 预算保留协议与判定带。

### 3.2 Anchor Recycling（核心新方法）
- 输入：10% 分层锚点 A（与语义层完全同锚，无额外标注）；
- 物理层低秩更新：LoRA(r=8) 注入 encoder_q/k 全部 Conv2d（B=0 初始化 = 起点等价冻结）；
- 新线性头（256→C）与 LoRA 联合训练，损失 = CE(A) + λ·(1−cos(f_s, f_t))
  （f_t = 冻结 pretext 特征，防遗忘锚）；
- BN running stats 冻结（训练/导出口径一致——TENT 诊断的教训）；
- 锚点增广：每 epoch 采样 n×V 个 Feeder_single 增广视图（shear 0.5 / pad 6），
  V 是按锚点质量选择的剂量；
- 适配后全模型冻结，重导特征，语义层按原协议重跑。
- 复杂度：可训练参数 3.2%（62,616/1,965,368），无新增标注。

### 3.3 统一测量协议
单实现 `run_arms()` 三臂（全量线性参照 / 10% 线性 / 10%+语义 ×10 seeds）、
判定带、坏串/数字保全门禁——所有数字同一把尺。

## 4. Experiments（✅ v0.3 草稿已成，见 experiments_section_draft.md）

### 4.1 Setup（✅）
### 4.2 零标注适配阴性套件（✅ Finding 1：TTA 74.08 / 同域 LoRA 74.99 / 跨域 71.98 / SupCon 74.98，全 ≤±1pp）
### 4.3 标注双用主结果（✅ Finding 2：NTU120 85.16 PARTIAL / PanAf 92.21 CONFIRMS / UCF 71.21 +20.1pp）
### 4.4 预算-分配率曲线（✅ Finding 3：5/10/15% 双用增益 +2.90/+3.91/+4.56pp 单调）
### 4.5 敏感性（✅ 仅 lr 与增广剂量显著；剂量响应单调 aug1/4/8 = 82.73/83.41/85.16，aug12 回落 82.47）
### 4.6 双侧边界（✅ 锚点质量下限：PanAf 40 锚零效应→增广恢复；校准门禁上限：lr4e-2 / PanAf lr≥1.5e-2 / aug12 触发 tau=0 硬失败）
### 4.7 效度威胁（✅ PanAf 量化/符号检验、UCF 弱参照、薄边际）

## 5. Discussion（✍️ 待扩写）

- 双用为何有效：k-ladder 已证"编码器质量由标签驱动"；双用把同一批标签的监督
  信号同时注入表征与决策两层，收益随域弱度与预算单调增强；
- 与 CIL 基线的关系：CIL（LwF 蒸馏）10% 档 68.6% 不支配解耦结构——双用是
  正交且更简单的机制；
- 与 full fine-tuning 的取舍：fullft 在合成域 98.2% 但摧毁标签效率叙事；
  双用保持 3.2% 可训练参数与冻结推理路径；
- 负结果的价值：四条零标注路线的系统性证伪本身构成可复用的证据。

## 6. Conclusion（✍️ 待扩写）

弱参照域的边界不是"冻结 vs 微调"的二选一，而是适配强度的双侧包络。Anchor
recycling 用零额外标注把两个弱参照域推过预注册判定线，并给出何时能推、推多远
的实用边界。

## 附录计划（✍️）

- A: 各 tier feeder/槽位映射（已有 supplementary 基础）
- B: E10 阴性套件全配置与日志指针
- C: E11/E12/E13/E14 逐种子明细
- D: 校准门禁失败案例（tau=0）分析

---

## 待办清单（下一轮）

1. 🟡 NTU120 aug8 配置的 lr3e-2/ep75 邻域已探（均触发门禁或回落）——峰值确认，无需再跑；
2. 🟡 UCF101 aug8 因超时未跑（aug4 已证低于无增广；aug8 预计更差，可放弃或改 2-seed 快查）；
3. ✍️ Intro/Related/Method/Discussion/Conclusion 依骨架扩写成文；
4. 🟡 CIL / ST-GCN 同预算对照补齐（审稿人视角的完备性）；
5. ✍️ 图：预算-分配率曲线图、剂量响应图、三 tier 主图（数据齐）。
