# 3. Method（英文框架稿 · P0.6）

> Owner: `docs/paper/method.md` · W5 窗口 2026-08-23 · 状态: **终稿候选 v0.3（2026-08-25 W36 终填轮）**——已归档数字终填、C7 换轨合规改写完成、"sliding-window baseline" 误标已按 W34 勘误修正；未执行项以显式 scope 声明保留（AimCLR++ 对比 / τ-K 敏感性全扫），不冒充已完成
> v0.2 对抗评审加固：§3.3 增迁移非平凡性论证段（防 incremental 指控）；§3.4 成本函数形式化（C_decoupled vs C_full @matched accuracy）。
> 写作规范（galaxy Method 章）: 以可复现为标准；呈现最终设计决策；消融留给 §4-§5；超参列表化。

---

This section presents the Physics–Semantics Decoupled framework (PSD). Section 3.1 fixes notation and gives an overview. Section 3.2 details the physics layer, which learns *how skeletons move* without any behavior labels. Section 3.3 details the semantic layer, which learns *what behaviors are called* from a small seed budget. Section 3.4 formalizes the decoupling mechanism that absorbs evolving evaluation criteria.

## 3.1 Problem Formulation and Overview

**Notation.** Let $X = \{x_1, \dots, x_T\}$ denote an unlabeled skeleton stream, where $x_t \in \mathbb{R}^{J \times 3}$ holds 3D joint coordinates for $J$ joints at frame $t$. A taxonomy $\mathcal{Y}$ maps behavior names to labels; evaluation criteria evolve by replacing $\mathcal{Y}$ with $\mathcal{Y}'$. The annotation budget is a set $S$ of seed clips with coarse labels produced by a rule engine over physical priors.

**Framework overview.** PSD factorizes recognition into two layers with a narrow interface:

- **Physics layer $\Phi$**: $\Phi$ maps skeleton windows to dynamics embeddings, trained once per species family on unlabeled streams. It comprises self-supervised pretraining (§3.2.1) and unsupervised temporal segmentation (§3.2.2).
- **Semantic layer $\Omega$**: $\Omega$ maps dynamics embeddings and segment proposals to labels under the current taxonomy $\mathcal{Y}$. It comprises anchor learning (§3.3.1), prototype clustering with iterated pseudo-labeling (§3.3.2), and semi-supervised self-training with an active-learning loop (§3.3.3).

The layers interact only through embeddings and proposals. This constraint is what allows $\mathcal{Y} \to \mathcal{Y}'$ transitions to leave $\Phi$ untouched (§3.4).

## 3.2 Physics Layer

### 3.2.1 Self-Supervised Pretraining

We adapt extreme-asymmetry contrastive pretraining [AimCLR, AAAI 2022; repo github.com/Levigty/AimCLR] to quadruped skeletons. Three adaptations matter. First, the joint graph is rebuilt for the target species topology rather than the human body plan; our current pipeline uses a 24-joint quadruped layout mapped into an NTU-compatible view via an identity correspondence plus one dead slot [P0.1 implementation detail]. Second, augmentation primitives are restricted to transformations that preserve quadruped plausibility. Third, the official weight initialization is deliberately skipped: we found that re-initializing all convolutional and linear weights to $\mathcal{N}(0, 0.02)$ collapses representations into a cone from which InfoNCE cannot escape on our data; skipping it restores normal convergence (diagnostic chain E1–E7 in `reports/p01-aimclr-2026-08-23.md`). We report this as a reproducibility note for practitioners reusing official code bases.

*Evidence.* On InterPet4D (226 SMAL-fitted sequences; 225 valid after one all-NaN exclusion), pretraining yields kNN top-1 of **20.89% versus an 8.33% random baseline (2.51×)** under a 5-fold protocol. This probe uses file-embedded subject identity as a proxy task and measures representation discriminability, not behavior accuracy; the metric caliber is disclosed to avoid cross-caliber confusion. （数据 owner：本段数字与 experiment-skeleton.md §E1 同源，回填与修订以 E1 为唯一维护点，此处只引用不独立改数。）[P0.1 report]

*Extended training schedules and the AimCLR++ backbone variant (77.2% NTU xsub reference) are outside the evaluation scope of the present version; both are recorded as candidate extensions in Section 4.4.*

### 3.2.2 Unsupervised Temporal Segmentation

Continuous skeleton streams must be cut into behavior-proposal units before labeling. We adopt motion-word quantization [SMQ, Gökay et al., ICCV 2025, arXiv 2508.04513]: local motion patterns are quantized into a discrete vocabulary, and segment boundaries are proposed where the quantized sequence changes regime.

*Evidence.* Segmentation quality is measured by boundary IoU against held-out annotations under a seeds pseudo-ground-truth protocol, compared against an equal-segment-count random-cut null baseline (Monte-Carlo expectation over repeated random cuts): **0.458 ± 0.049** versus ≈0.30 (≈1.53×), with boundary F1 of 0.343 reported alongside (`reports/p02-smq-iou-eC-seeds-recheck.json`, public-real tier). A sliding-window comparison arm is not part of the present evaluation; its design is registered for future ablation (Section 5). （数据 owner：experiment-skeleton.md §E2，此处只引用不独立改数；W34 勘误——对照基线系随机切分 null 而非滑窗，成稿禁写 "vs sliding window"。）

## 3.3 Semantic Layer

The semantic layer grows a complete taxonomy from the rule-engine seed set $S$, following the anchor–cluster–pseudo-label recipe transferred from image-domain few-shot recognition [W1 novelty survey; see §2.3].

The transfer is not a drop-in replacement, and its non-triviality is a designed-in property rather than an incidental one. Three domain shifts require re-engineering rather than re-application: (i) anchors must be defined over *temporal segment embeddings* from a self-supervised physics encoder instead of image features, so anchor quality depends on representation adequacy (§3.2.1); (ii) prototype clustering operates on variable-length proposals delivered by unsupervised segmentation rather than fixed-size inputs, coupling label expansion quality to segmentation quality (§3.2.2); and (iii) confidence calibration must tolerate quadruped kinematics whose intra-class motion variance differs from both image textures and human locomotion. Each shift is validated separately in §4–§5 before the composed pipeline is evaluated.

### 3.3.1 Anchor Learning

Rule-engine coarse labels over physical priors provide seed anchors $A = \{(e_i, y_i)\}$, where $e_i$ are physics-layer embeddings of seed segments. Anchors are learned as class-representative prototypes initialized from seed embeddings and refined during training. Unlike anchor-based contrastive objectives that treat anchors as sample pairs [MAC-Learning, TPAMI 2022], our anchors carry semantic identity and directly supervise cluster assignment.

### 3.3.2 Prototype Clustering and Iterated Pseudo-Labeling

Unlabeled proposals from §3.2.2 are embedded by $\Phi$ and assigned to their nearest prototype with confidence $\kappa$. Proposals above threshold $\tau$ receive pseudo-labels and join the training pool; prototypes are then re-estimated. Algorithm 1 summarizes the loop.

```text
Algorithm 1: Anchor-guided iterative pseudo-labeling
Input : physics encoder Φ (frozen), seeds A, unlabeled proposals U
Output: classifier Ω under taxonomy Y
1: initialize prototypes P from seed anchors A
2: repeat
3:   assign each u ∈ U to nearest prototype p_k, confidence κ(u)
4:   L_high ← {u : κ(u) ≥ τ};  add pseudo-labeled pairs to pool
5:   update Ω on seeds ∪ pseudo-labeled pool
6:   re-estimate P from embedded features
7: until proposal assignment stabilizes or iteration budget reached
```

Design decisions fixed at this stage: confidence filtering uses prototype-margin scores; each iteration retrains only $\Omega$, never $\Phi$; the iteration budget guards against confirmation bias accumulating across rounds; and seed-noise amplification is bounded by treating rule-engine seeds as high-recall/low-precision priors—anchors initialized from noisy seeds are corrected by confidence-filtered cluster consensus and verified in downstream evaluation rather than trusted as ground truth. Class imbalance in animal behavior distributions (long-tail rare behaviors) is handled at the prototype level via frequency-aware margin thresholds rather than resampling; this choice is ablated in §5.

*Evidence.* Anchor-guided prototype clustering reaches purity **0.5339** versus a 0.3306 random-assignment baseline (**1.62×**) with three run-seeds returning identical values; injecting 30% label noise into the seeds degrades purity by only 3.1 pp (to 0.5025±0.0063, still 1.52× baseline), and enlarging the seed ratio from 25% to 100% yields ≤1 pp additional purity gain—prototype quality saturates early (`reports/p03-jia-phasea-results.json`, public-real tier). Full sensitivity sweeps over $\tau$ and $K$ are registered for Section 5.

### 3.3.3 Semi-Supervised Self-Training and Small-Budget Initialization

Pseudo-labeled coverage is consolidated by temporal contrastive self-training in the spirit of TCL [Singh et al., CVPR 2021, arXiv 2102.02751]（R8 勘误 2026-09-04：原记载『82.7%@10% vs 88.6% 全监督』经原文 PDF 全文检索证伪——该文无 NTU 实验、无此数字，其数据集为 Mini-Something-V2/Jester/Kinetics-400/Charades-Ego；方法精神引用保留，数字删除）。 To make very small annotation budgets usable, a warm-start protocol initializes this stage from previously trained semantic-layer weights before expansion proceeds within a budget of 100–200 clips.

*Evidence.* Under distribution shift on the synthetic-offset tier, the warm-started semantic layer reaches **82.0% top-1 on 22 classes from only 20 labeled clips** (mean over three seeds, ±4.3; cold-start protocols remain near chance at this budget under their respective distributions), with uniform expansion reaching 95.7% at a 200-clip ceiling (`reports/p05-al-efficiency-warmstart-short-2026-08-25.json`). Uncertainty-based sampling strategies are evaluated separately as an exploratory negative finding in Section 5 and carry no methodological claim here; three-tier main results are reported in Section 4 (`reports/p05-public-real-partialclass-result-2026-08-25.json` for the public-real partial caliber). （数据 owner：experiment-skeleton.md §E4/E5，此处只引用不独立改数；W23 comparability 条款——warm-start 与冷启动绝对数值禁止同协议并列。）

## 3.4 Decoupling Mechanism

We formalize an evaluation-criteria evolution as a taxonomy transition $\mathcal{Y} \to \mathcal{Y}'$ (splitting, merging, renaming, or adding behaviors). We define the transition cost as $C(\mathcal{Y} \to \mathcal{Y}') = $ human annotation units consumed plus wall-clock retraining time, measured at matched final accuracy. Under decoupling:

1. the physics layer $\Phi$ and its pretrained parameters remain frozen;
2. existing segment proposals are re-used—only their label assignments change;
3. the semantic layer is rebuilt through §3.3 using seeds under $\mathcal{Y}'$;
4. the reported quantity is $C_{\text{decoupled}}(\mathcal{Y} \to \mathcal{Y}')$ versus $C_{\text{full}}(\mathcal{Y} \to \mathcal{Y}')$, where the full baseline retrains representation learning under $\mathcal{Y}'$.

We hypothesize that $C_{\text{decoupled}} < C_{\text{full}}$ at matched accuracy, and that accuracy under $\mathcal{Y}'$ matches full retraining within noise.

*Evidence.* Both hypotheses are supported on the synthetic tier: the decoupled semantic-layer rebuild averages **31.1 s** of wall-clock retraining versus **188.7 s** for full-pipeline retraining on the full tier (clean GPU path, cuda+AMP; small tier ratio 7.32×), a **6.07×** reduction with same-seed paired ratios never below 4.00× — we state a conservative **≥3×** bound in claims; accuracy is statistically equivalent (decoupled −0.91 pp on the full tier: 95.76% vs 96.67%; small tier +2.27 pp — both within the <2.3 pp noise band given validation sizes); annotation units are identical across arms by construction (1760 train / 440 val each), so the saving is computational rather than labeling-driven (`reports/c1-decouple-cost-full-2026-08-25.json` aggregated + `reports/c1-decouple-cost-2026-08-24.md` §2+§9).

---

## 超参与复现清单（当前已冻结项，供 §4.2 引用）

| 项 | 值 | 来源 |
|----|----|------|
| 预训练 epoch / lr / queue | 120 / 0.05 / 1024（首版可用配置，未做系统搜索——诚实披露） | `reports/p01-aimclr-2026-08-23.md` |
| 窗口重采样 | T=64 | 同上 |
| 有效训练序列 | 225（226 剔 1 全 NaN） | 同上 |
| GPU | RTX 5060 Laptop 8GB，单卡 | HANDOVER §4 |
| 其余超参（τ、K、迭代轮数、TCL 权重） | ⏳ 待 P0.3/P0.4 冻结后回填本表 | — |

## 自审记录

| 检查项 | 结果 |
|--------|------|
| 可复现性：设计决策 + 已冻结超参列出 | ✅ 表格化；未冻结项显式标注 |
| 消融内容不混入 Method | ✅ 敏感性全部指向 §4/§5 |
| 主张与证据分离 | ✅ §3.2.1/§3.2.2/§3.3.2/§3.3.3/§3.4 五处 Evidence 块均已按归档报告终填（W36）；AimCLR++ 对比与 τ-K 全扫以显式 scope 声明保留（未执行不冒充） |
| 口径诚实 | ✅ dog-ID 代理探针口径披露；坍缩事故写成复现性注记而非隐藏；warm-start 与冷启动禁止同协议并列（comparability 护栏）；C1 臂属方向经当次 JSON 复核（decouple 31.1s vs baseline 188.7s） |
| C7 换轨合规 | ✅ §3.3.2 "active-learning loop 验证"改"downstream evaluation"；§3.3.3 标题与正文换轨 warm-start，AL 仅以探索性发现出现；无 ≥85% 预算主张残留 |
| 引用纪律 | ✅ SMQ/TCL 题录 W17 补全；TCL 数字 82.7%/88.6% 已于 R8（2026-09-04）原文 PDF 证伪并全链删除——该待办闭环 |

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 窗口框架稿：四小节结构 + 算法 1 伪代码 + 占位符体系 |
| v0.2 | 2026-08-23 | 对抗评审加固：迁移非平凡性论证段 + 解耦成本函数形式化（matched accuracy 条件显式化） |
| v0.3 | 2026-08-25 | **W36 终填轮**：§3.2.2 证据块终填（IoU 0.458±0.049 vs ≈0.30 null + F1 0.343）并按 W34 勘误修正 "sliding-window baseline" 误标为等段数随机切分 null；§3.3.2 证据块终填（purity 0.5339/1.62×、噪声 30% 仅降 3.1pp、种子比饱和 ≤1pp）+ AL 验证措辞合规化；§3.3.3 换轨定稿（标题改 Small-Budget Initialization、82.0%@20 clips 入证据块、AL 负结果指向 §5、三层主结果指针）；§3.4 [CLAIM NEEDS EVIDENCE] → 证据块（6.07×/31.1s vs 188.7s/同 seed 最小 4.00×/统计等效 −0.91pp/+2.27pp/标注单元打平 1760/440）；§3.2.1 尾 PENDING 改显式 scope 声明（AimCLR++ 与延长调度不在本版范围）；头部状态与自审记录同步 |
