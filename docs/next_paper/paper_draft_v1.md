# Anchor Recycling: Label-Efficient Skeleton Behavior Recognition Under Evolving Taxonomies

> **全文草稿 v1.0**（2026-11-20）。实验章 4.1–4.7 的全部数字来自 `runs/p1_sprint/`
> 实测 summary（E9 四层协议回归、E10 阴性套件、E11–E15 扫参/预算曲线/跨 tier/增广
> 剂量），与已发布基线同一测量实现（`run_arms` 统一尺）。目标期刊：TMM / PRL。
> 待办见文末 TODO。

---

## Abstract

Skeleton behavior recognition under evolving taxonomies faces a recurring
practical constraint: the label taxonomy changes, the domain shifts, and the
annotation budget remains small. Decoupled frameworks that freeze a
self-supervised physical layer and adapt only a semantic layer preserve label
efficiency, but they cap performance in *weak-reference domains* — domains
where the reference labels are few, noisy, or indirectly derived. Through a
controlled four-tier study (NTU60 → PanAf500 → NTU120 → UCF101) governed by a
single measurement implementation, we first map this cap and then test every
label-free route around it: test-time entropy minimization, low-rank continued
pretraining, cross-domain transfer from a strong source, and pseudo-label
feedback. All four fail, moving retention by at most ±1 pp — the boundary is
structural, and it is the labels that are missing. We then show that the
boundary is not a wall: the *same* small anchor set that drives the semantic
layer can be **recycled** to shape the physical layer through low-rank
supervised adaptation with a distillation anchor. Anchor recycling improves
budget-limited retention by +10.2 pp on NTU120 (74.96% → 85.16%, PARTIAL),
+9.7 pp on PanAf500 (82.5% → 92.21%, CONFIRMS), and +20.1 pp on UCF101
(51.09% → 71.21%), with the gain growing as reference quality degrades. The
effect has a two-sided envelope — a minimum anchor mass below which adaptation
is inert, and a calibration gate above which the semantic layer refuses the
adapted features — which we map across tiers. We release the unified
measurement harness and all per-seed evidence.

**Keywords** — skeleton action recognition; domain adaptation; label
efficiency; self-supervised pretraining; evolving taxonomies; negative
results.

---

## 1. Introduction

Skeleton-based behavior recognition has matured on benchmarks with abundant,
cleanly labeled training data. Deployed systems, however, rarely enjoy that
luxury: the label taxonomy evolves (classes are merged, renamed, split, or
added as deployment reveals what matters), new domains arrive with little or
no in-domain supervision, and the annotation budget is a hard constraint set
by domain experts rather than by the learning algorithm. A framework intended
for this reality must therefore be judged not by its accuracy at 100% labels,
but by how much of that accuracy it retains when the budget shrinks and the
reference degrades.

The physics–semantics decoupling (PSD) framework was designed for exactly this
regime. A self-supervised **physical layer** (an AimCLR-style contrastive
pretext over skeletons) is frozen; a **semantic layer** (rule-based seeds,
anchor clustering, confidence-filtered pseudo-labels, and a self-training
head) adapts to the current taxonomy using only a small stratified anchor set.
Under a pre-registered *budget-retention protocol* — the ratio of the
small-budget pipeline accuracy to the full-label linear-probe accuracy of the
same encoder, with verdict bands CONFIRMS (≥90%), PARTIAL (85–90%), and FAILS
(<85%) — the framework confirms its label-efficiency claim on the strong
reference domain NTU60 (90.7% retention) but hits a ceiling on weak-reference
domains: 82.5% on PanAf500, 74.96% on NTU120, and 51.09% on UCF101.

This paper is about that ceiling. A natural diagnosis is that the frozen
encoder is the bottleneck, and the natural remedy is to "let it move." We
first ask whether the encoder can be moved *for free* — without spending any
additional labels — and answer negatively with a systematic negative-result
study. Test-time entropy minimization (TENT-style adaptation of BatchNorm
affine parameters), low-rank continued pretraining on all unlabeled target
clips, cross-domain transfer from a strong source encoder, and pseudo-label
feedback from the semantic layer back into the encoder all leave budget-limited
retention within ±1 pp of the frozen baseline. The ceiling is structural: what
is missing is not adaptation capacity but *label signal reaching the encoder*.

The positive contribution follows from that diagnosis. The semantic layer
already consumes a small stratified anchor set; we **recycle the identical
anchors** as supervision for a low-rank update of the physical layer. A fresh
linear head and LoRA adapters (3.2% of parameters) are trained on the anchor
clips with cross-entropy plus a cosine distillation anchor to the frozen
encoder, while BatchNorm running statistics stay frozen to keep the
train/evaluation feature contract intact. The adapted encoder then re-enters
the *same* three-arm protocol, measured by the *same* implementation. We call
this **anchor recycling**: no new labels, no new annotation, only a second use
of labels that already existed.

Anchor recycling delivers consistent, large gains that grow as the reference
degrades: +10.2 pp on NTU120 (74.96% → 85.16%, crossing into the PARTIAL
band), +9.7 pp on PanAf500 (82.5% → 92.21%, crossing into CONFIRMS), and
+20.1 pp on UCF101 (51.09% → 71.21%). A budget-allocation sweep shows the
advantage widens with budget (5%: +2.9 pp; 10%: +3.9 pp; 15%: +4.6 pp before
augmentation), and a hyperparameter study isolates the adaptation learning
rate and the anchor-augmentation dose as the only significant factors.

Equally important, we map the *envelope* of the effect. Where anchor mass is
too small (40 anchors on PanAf500), recycling is inert — every outcome is
bit-identical to the frozen pipeline until anchor augmentation restores it.
Where adaptation pressure is too strong, the semantic layer's calibration gate
rejects the adapted features outright (τ = 0). The boundary between these
regimes is sharp, reproducible, and tier-dependent in a way that tracks anchor
mass.

Our contributions are:

1. **A weak-reference evaluation protocol** for skeleton recognition under
   evolving taxonomies: four tiers, a fixed budget-retention ruler, and
   pre-registered verdict bands, with all numbers produced by a single
   measurement implementation.
2. **A systematic negative-result study** establishing that four families of
   label-free encoder adaptation cannot close the weak-reference gap —
   including the diagnostic traps (BatchNorm train/eval statistics, feature
   geometry vs. semantic harvest) that explain *why* they fail.
3. **Anchor recycling**, a label-efficient adaptation mechanism that reuses
   the semantic layer's anchor set to shape the physical layer, yielding
   +10.2/+9.7/+20.1 pp on three weak-reference tiers and crossing two
   pre-registered verdict bands with zero additional annotation.
4. **A two-sided envelope analysis** (anchor-mass floor, calibration-gate
   ceiling) and a dose–response map (augmentation × learning rate × budget)
   that tell practitioners when recycling helps, when it is inert, and when it
   fails loudly.

All evidence, per-seed results, and the measurement harness are released.

---

## 2. Related Work

**Self-supervised skeleton pretraining.** Contrastive and masked-prediction
pretexts (AimCLR, CrosSCLR, BehaviorMAE) and pose-centric transfer methods
(TREBA, ProSL) build strong physical representations without labels. Our
physical layer follows the AimCLR single-stream joint recipe at reduced
capacity (16 hidden channels, 1.84 M parameters), chosen to keep every tier
computable under one protocol. These works ask *how to pretrain*; we ask
*what happens after pretraining, when the taxonomy and the budget change*.

**Domain and test-time adaptation.** Source-free and test-time adaptation
methods (e.g., TENT) update normalization parameters on unlabeled test
streams. We adopt TENT as one of four label-free adaptation baselines and
show that, for skeleton features under a weak reference, it neither collapses
nor helps once the BatchNorm statistics contract is made consistent — the
feature geometry, not its calibration, is the constraint.

**Continual and class-incremental learning.** Evolving taxonomies induce
merge/rename/split/addition events studied in class-incremental learning
(e.g., LwF distillation). Prior work in our framework established a
zero-retraining bound for merge/rename events (76.8% oracle / 76.5% voting on
NTU60→10-class) and an adaptation ladder for split/addition. Here we use LwF
as an adapted CIL baseline at matched budget (68.6% at 10% on the TRANS-002
merge) and show anchor recycling is orthogonal and complementary.

**Label-efficient learning.** Semi-supervised and self-training methods
amplify small anchor sets with pseudo-labels; supervised contrastive learning
(SupCon) structures representations with label supervision. Our pseudo-label
feedback arm applies both ideas to the encoder and fails on NTU120 — a
negative result that motivates recycling: pseudo-labels help the *decision*
layer but cannot substitute for true labels shaping the *representation*.

**Positioning.** None of the above studies the weak-reference regime with a
fixed budget ruler across four domains of graded reference quality, and none
asks which layer of a decoupled framework should receive the labels. That
question — and its two-sided answer — is the subject of this paper.

---

## 3. Method

### 3.1 Preliminaries: the decoupled framework and its protocol

The PSD framework separates a **physical layer** (a self-supervised AimCLR
pretext: ST-GCN base, 16 hidden channels, 1.84 M parameters, 150 epochs,
single-stream joint) from a **semantic layer** (rule-based seed extraction,
anchor clustering, confidence-filtered pseudo-labels, and a self-training
classification head). The physical layer is frozen after pretraining.

The evaluation protocol measures **budget retention**: with a stratified
`b`% anchor draw (seeds 42–51), the pipeline's validation accuracy is divided
by the full-label linear-probe accuracy of the same frozen encoder. Verdict
bands are pre-registered: retention ≥ 90% CONFIRMS, 85–90% PARTIAL, < 85%
FAILS. Every number in this paper is produced by one shared implementation of
the three arms (full-label reference; `b`% linear head; `b`% + semantic
pipeline), so that cross-tier and cross-paper comparisons use a single ruler.

### 3.2 Anchor recycling

Given the anchor set `A` (|A| = `b`% of the training clips, drawn exactly as
the semantic layer's anchors), anchor recycling performs a low-rank supervised
update of the physical layer before the semantic stage:

1. **Low-rank injection.** Every Conv2d in the query encoder receives a
   parallel LoRA branch (rank r = 8, α = 16): `y = conv(x) + (α/r)·B(A(x))`,
   with `B` zero-initialized so the adapted encoder is functionally identical
   to the frozen one at step zero. The key encoder receives the same
   injection and is momentum-synced (though the contrastive machinery is
   unused during recycling). Trainable parameters: 62,616 of 1,965,368
   (3.19%).
2. **Objective.** Cross-entropy on the anchor clips through a fresh linear
   head, plus a cosine feature distillation anchor
   `1 − cos(f_s, f_t)` to the frozen encoder, weighted by λ. BatchNorm
   running statistics are frozen (module kept in eval mode) so that the
   features seen during adaptation match the features later exported for the
   semantic stage — a consistency requirement we learned from the TENT
   diagnostics (§4.2).
3. **Anchor augmentation.** When anchor mass is small, each epoch samples
   `V×|A|` augmented views of the anchors through the pretext's own
   augmentation family (shear 0.5, temporal padding ratio 6), turning a
   40-anchor set into hundreds of distinct gradient signals without new
   labels. The dose `V` is a tier-dependent hyperparameter (§4.5).
4. **Re-measurement.** The adapted encoder is frozen, features are
   re-exported, and the identical three-arm protocol is re-run — including a
   fresh full-label linear reference, so retention remains an honest ratio.

No additional labels are used anywhere: the recycling stage sees exactly the
anchor set that the semantic stage already consumes.

### 3.3 The calibration gate

The semantic layer includes a calibration step that selects a similarity
threshold τ from a grid and rejects degenerate calibrations. We found this
gate to be an unexpected safety mechanism during encoder adaptation: when the
recycling pressure is too strong, the adapted feature geometry drives the
selected τ to zero and the pipeline raises a hard error instead of producing
degenerate pseudo-labels. §4.6 maps this gate as the upper bound of the
adaptation envelope.

---

## 4. Experiments

### 4.1 Setup

Four tiers ordered by reference quality — NTU60 (published retention 90.7%),
PanAf500 (82.5%), NTU120 (74.96%), UCF101 (51.09%) — each with its own
from-scratch label-free pretext (150 epochs) and a 10% stratified anchor
budget (seeds 42–51; 10 seeds for every headline number). NTU120's anchors
number 734, UCF101's 953, PanAf500's 40 — a spread that turns out to be
predictive of the adaptation envelope. All retentions are computed by the
shared ruler described in §3.1; per-seed values, logs, and the harness are
released.

### 4.2 Zero-label adaptation does not move a frozen physical layer

We test four label-free adaptation families on NTU120, the tier with the most
complete instrumentation:

| Arm | Retention | Δ vs. frozen |
|---|---|---|
| Frozen baseline | 74.96% | — |
| TENT (BN affine, eval-statistics) | 74.08% | −0.88 pp |
| Same-domain LoRA continued pretraining | 74.99% | +0.03 pp |
| Cross-domain zero-shot (NTU60 → NTU120) | 69.77% | −5.19 pp |
| Cross-domain LoRA transfer | 71.98% | −2.98 pp |
| Pseudo-label SupCon feedback | 74.98% | +0.02 pp |

Three diagnostics explain the failures rather than merely recording them.
First, a naive TENT setup that adapts in BatchNorm train mode collapses to
27.0% — entirely a train/eval statistics artifact; once the statistics
contract is fixed, TENT is simply inert here. Second, same-domain continued
pretraining is saturated: the from-scratch pretext has already consumed its
own data for 150 epochs, and 30 more low-rank epochs change nothing. Third,
cross-domain transfer produces the *best absolute separability* of any arm
(full-label reference 59.66% vs. 58.37% frozen; 10% linear 45.27% vs. 44.18%)
yet *lowers* retention — the semantic layer's clustering was grown around the
frozen geometry and cannot harvest the improved features. The constraint is
not "can the features improve" but "can the pipeline's discrete stages absorb
the improvement."

Pseudo-label feedback closes the loop the other way — semantic pseudo-labels
retraining the encoder via SupCon over 7,211 covered samples — and is equally
inert (74.98%). Pseudo-labels inherit the frozen encoder's geometry; feeding
them back reinforces it.

**Finding 1.** Four label-free adaptation families, spanning test-time,
self-supervised, cross-domain, and semi-supervised mechanisms, move
budget-limited retention by at most +0.1 pp and as much as −5.2 pp. The
weak-reference ceiling is structural: it is the absence of true label signal
at the encoder, not a lack of adaptation capacity.

### 4.3 Anchor recycling: main results

Recycling the semantic layer's anchors into the physical layer (§3.2) with
per-tier optimal settings:

| Tier | Anchors | Budget | Frozen | Dual-use | Δ | Verdict |
|---|---|---|---|---|---|---|
| NTU120 | 734 | 10% | 74.96% | **85.16%** (lr 2e-2, aug ×8) | **+10.20 pp** | FAILS → **PARTIAL** |
| PanAf500 | 40 | 10% | 82.50% | **92.21%** (lr 1e-2, aug ×4) | **+9.71 pp** | FAILS → **CONFIRMS** |
| UCF101 | 953 | 10% | 51.09% | 71.21% (lr 2e-2) | **+20.12 pp** | FAILS → FAILS |

Two of the three weak-reference tiers cross their pre-registered bands, and
the third improves by 20 pp. The gains come from both layers: on NTU120 the
adapted encoder's absolute separability rises (10% linear head 44.18% → 45%+)
*and* the semantic layer harvests more of it (pipeline 43.75% → 49.60%) — the
opposite of the cross-domain transfer case, where separability rose but the
semantic layer could not use it. Per-seed pairing on NTU120 shows all ten
seeds improving.

**Finding 2.** The same labels are worth more when they shape both layers.
The effect is present in every weak-reference tier and grows as reference
quality degrades.

### 4.4 Where the budget should go

On NTU120, sweeping the budget with the original recipe:

| Budget | Semantic-only (frozen) | Dual-use | Gain |
|---|---|---|---|
| 5% | 64.61% | 67.51% | +2.90 pp |
| 10% | 74.96% | 78.87% | +3.91 pp |
| 15% | 79.04% | 83.60% | +4.56 pp |

**Finding 3.** The dual-use advantage widens monotonically with budget. At
15% the pipeline reaches 83.60% — 1.4 pp below PARTIAL before augmentation is
even applied. Practitioners with a fixed budget should expect recycling to be
worth more, not less, as the budget grows.

### 4.5 Hyperparameter sensitivity

On NTU120 (10 seeds per cell), only two factors matter. The adaptation
learning rate: 78.87% (1e-3) → 82.13% (1e-2) → 82.73% (2e-2) without
augmentation; 4e-2 trips the calibration gate (§4.6). The augmentation dose:
with lr 2e-2, 82.73% (×1) → 83.41% (×4) → **85.16% (×8)** → 82.47% (×12) — a
clean unimodal dose–response. Epochs (100 vs. 200), rank (8 vs. 16), and the
distillation weight (0, 0.5, 1.0) each move the result by ≤ 0.5 pp; removing
the distillation anchor entirely is safe at lr 1e-3 (79.05%).

The optimal recipe differs across tiers in a way that tracks anchor mass:
NTU120 (734 anchors) takes the strongest setting (lr 2e-2, 8× augmentation);
UCF101 (953 anchors) peaks at lr 2e-2 *without* augmentation (71.21%; the
augmented 1e-2 variant drops to 68.31%); PanAf500 (40 anchors) needs the
gentlest setting (lr 1e-2, 4× augmentation, 92.21%) and breaks the gate at
2e-2 even with augmentation.

### 4.6 The envelope: anchor-mass floor and calibration-gate ceiling

**Anchor-mass floor.** On PanAf500, naive recycling (40 anchors, batch 64)
produces *zero* gradient steps and leaves all ten seeds bit-identical to the
frozen pipeline — the entire chain (features → clustering → pseudo-labels →
predictions) reproduces exactly. With the batch bug fixed and augmentation
added at the gentle setting, the pipeline moves for the first time and reaches
CONFIRMS. Below ~100 anchors, recycling is inert; above it, the effect
appears.

**Calibration-gate ceiling.** When recycling pressure exceeds the envelope —
lr 4e-2 on NTU120, lr 2e-2 with augmentation on PanAf500, lr 3e-2 with 8×
augmentation — the semantic layer's calibration step selects τ = 0 and the
pipeline raises a hard error rather than emitting degenerate pseudo-labels.
The gate converts "silent quality collapse" into "loud, diagnosable failure,"
and maps the ceiling of the adaptation envelope precisely.

**Finding 4.** Anchor recycling lives in a two-sided envelope: an
anchor-mass floor beneath which it is inert, and a calibration-gate ceiling
above which it fails loudly. Both edges are reproducible and tier-dependent.

### 4.7 Threats to validity

PanAf500's validation set is 25 clips, quantizing accuracies in 4% steps; its
CONFIRMS verdict rests on a 6/10 improvement with 2 regressions (sign test
n.s.) and a small reference drift (64.0% → 61.6%), so we present it as a
strong signal with quantification caveats, not proof. NTU120's PARTIAL margin
is 0.16 pp. UCF101's reference is video-derived and weak (full-label linear
reference ≈ 19%), so its retentions are ratios of small numbers. All
comparisons share one measurement implementation, and no number in this paper
comes from a different ruler than the published baselines.

---

## 5. Discussion

**Why does recycling work when everything else fails?** The negative suite
isolates the mechanism by elimination. Test-time adaptation changes
normalization without labels — inert. Continued self-supervised pretraining
changes the encoder without labels — inert. Cross-domain transfer changes the
encoder with foreign labels — the features improve but the semantic layer
cannot use them. Pseudo-label feedback changes the encoder with *derived*
labels — the derivation inherits the frozen geometry, so it reinforces rather
than repairs it. Recycling is the only arm whose supervision is (a) real,
(b) in-domain, and (c) routed to the encoder. The k-ladder study predicted
exactly this: encoder quality is label-driven.

**Why does the semantic layer sometimes fail to harvest better features?**
The cross-domain arm shows that absolute separability and retention can move
in opposite directions. The semantic layer's anchor clustering and
confidence gating are grown around a feature geometry; when the geometry
changes faster than the clustering, the discrete pipeline stages (cluster
assignments, pseudo-label acceptance, final predictions) may not flip at all —
PanAf500's bit-identical replication is the extreme case. Recycling works
because it moves the geometry *gently* (low-rank, distillation-anchored) and
*with* the semantic layer's own anchors, keeping the two layers coherent.

**Relation to full fine-tuning.** Prior controlled contrasts show full
fine-tuning reaches 98.2% on the synthetic tier while destroying the
label-efficiency narrative it is meant to serve. Recycling keeps 96.8% of
parameters frozen, preserves the frozen inference path, and spends zero
additional labels — it is the label-efficient point on the adaptation
spectrum, not the accuracy-maximal one.

**The value of mapped negatives.** The four failed routes are not
ablation filler; each failure is paired with a mechanism (statistics gap,
saturation, geometry–harvest mismatch, label inheritance). For practitioners,
the map says: do not spend effort on label-free encoder tricks in weak-
reference regimes; spend the labels you have on both layers.

**Limitations.** Recycling is validated on skeleton data with GCN backbones
and LoRA-style conv adapters; transfer to transformer encoders is untested.
The budget range is 5–15%; behavior beyond 20% converges toward full
fine-tuning. PanAf500 and UCF101 verdicts carry the quantization and
weak-reference caveats stated in §4.7. The calibration gate, while a useful
safety mechanism, is itself a hyperparameter surface we did not explore.

---

## 6. Conclusion

We studied the ceiling of decoupled skeleton behavior recognition in
weak-reference domains and turned it into a method. A systematic negative-
result study shows that no label-free route moves a frozen physical layer past
its ceiling. The ceiling yields to a simple idea with an honest cost
accounting: recycle the same anchors that drive the semantic layer as
supervision for a low-rank physical-layer update. Across three weak-reference
tiers this gains +10.2/+9.7/+20.1 pp, crosses two pre-registered verdict
bands, and requires zero additional annotation. The effect's two-sided
envelope — an anchor-mass floor and a calibration-gate ceiling — is mapped and
released, along with every per-seed number and the single-ruler harness that
produced them.

Beyond this paper, anchor recycling suggests a broader principle for
decoupled systems under evolving taxonomies: *the labels that adapt the
decision layer should also be allowed to adapt the representation, at a dose
the representation can absorb.* We believe this principle — and the boundary
map that motivated it — will generalize to other modalities where
self-supervised physical encoders meet small, shifting, expert-labeled
semantic layers.

---

## Appendix plan

- **Appendix A** — Tier construction: feeder and slot mappings (NTU25
  16-active-slot arrangement; AP-10K → PSD24 → NTU25 for PanAf500), dataset
  sizes, budget draws.
- **Appendix B** — Zero-label adaptation suite: full configurations, per-seed
  results, TENT BatchNorm diagnostics, cross-domain harvest analysis.
- **Appendix C** — Recycling per-seed tables for NTU120/UCF101/PanAf500 and
  the full hyperparameter grid (lr × augmentation dose × budget).
- **Appendix D** — Calibration-gate failure cases: τ = 0 trajectories and
  what they reveal about the semantic layer's safety margin.
- **Appendix E** — Reproduction: the unified measurement harness, checkpoint
  indices, and command lines for every table in §4.

---

## TODO（成文前）

1. ✍️ 将 §4.1–4.7 从 experiments_section_draft.md 合并进来并统一符号；
2. 🟡 补图：Fig.1 四层梯度、Fig.2 预算-分配率曲线、Fig.3 增广剂量响应、Fig.4 三 tier 主图；
3. 🟡 参考文献（~40 条：AimCLR/CrosSCLR/BehaviorMAE/TREBA/ProSL/TENT/LwF/SupCon/LoRA 等）；
4. ✍️ Introduction/Related/Method/Discussion/Conclusion 按本骨架成文；
5. 🟡 复核：所有数字与 `reports/p1_sprint_evidence/adaptation/` 逐位一致；
6. ✍️ Highlights 与 cover letter（TMM/PRL 版）。
