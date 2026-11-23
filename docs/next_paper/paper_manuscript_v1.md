# Anchor Recycling: Label-Efficient Skeleton Behavior Recognition Under Evolving Taxonomies

> **Manuscript draft v1.0** — 2026-11-20. Prose draft for TMM/PRL submission.
> Editorial notes and pre-submission TODO:
> `paper_manuscript_v1_editorial_notes.md`.

---

## Abstract

Skeleton behavior recognition under evolving taxonomies faces a recurring
constraint: label taxonomies and domains keep changing while annotation
budgets stay small. Decoupled frameworks that freeze a self-supervised physical layer and
adapt only a semantic layer preserve label efficiency, but they cap
performance in *weak-reference tiers* where labels are few, noisy, or
indirectly derived. Through a controlled four-tier study (NTU60 → PanAf500 →
NTU120 → UCF101) governed by a single measurement implementation, we first map
this ceiling and then, on the NTU120 tier, test four label-free routes around
it: test-time entropy minimization, low-rank continued pretraining,
cross-domain transfer from a strong source, and pseudo-label feedback. All
four fail: no route gains more than +0.1 pp, and none of them supplies label
signal to the encoder. We then show that the ceiling can be raised: the *same* small anchor set
that drives the semantic layer can be **recycled** to shape the physical layer
through low-rank supervised adaptation with a distillation anchor. Anchor
recycling improves budget-limited retention by +10.20 pp on NTU120 (74.96% →
85.16%, PARTIAL), +9.71 pp on PanAf500 (82.50% → 92.21%, CONFIRMS), and +20.12
pp on UCF101 (51.09% → 71.21%), the largest gain appearing on the
weakest-reference tier. The effect has a two-sided envelope: a minimum anchor mass below
which unaugmented adaptation is inert, and a calibration gate above which the semantic
layer refuses the adapted features. We map both axes across tiers, with a
unimodal response to augmentation peaking at 85.16%. We release the unified
measurement harness and all per-seed evidence.

---

## 1. Introduction

Skeleton-based action recognition has matured on benchmarks with abundant,
clean labels. Deployed systems rarely enjoy that luxury. The label taxonomy
itself evolves (classes are merged, renamed, split, or added as a deployment
reveals what actually matters), and new domains arrive with little in-domain
supervision because the annotations must be produced by scarce domain
experts. A recognition framework intended for this reality must therefore be
judged by how much of its full-label accuracy it *retains* as the budget
shrinks and the reference degrades.

The physics–semantics decoupling (PSD) framework was designed for exactly
this regime. A self-supervised **physical layer** (an AimCLR-style [1]
contrastive pretext over skeletons) is frozen after pretraining; a
**semantic layer** (rule-based seed extraction, anchor clustering,
confidence-filtered pseudo-labels, and a self-training head) adapts to the
current taxonomy using only a small stratified anchor set. The framework is judged by a
pre-registered *budget-retention protocol*: the ratio of the small-budget
pipeline accuracy to the full-label linear-probe accuracy of the same frozen
encoder, with verdict bands CONFIRMS (≥ 90%), PARTIAL (85–90%), and FAILS
(< 85%). It confirms its label-efficiency claim on the strong-reference
tier NTU60 (90.7% retention at a 10% budget). On weak-reference
tiers, however, it hits a ceiling: 82.50% on the chimpanzee behavior tier
PanAf500, 74.96% on the 60-class NTU120 subset, and 51.09% on UCF101
skeletons.

This paper is about that ceiling, and about the natural hypothesis that
adapting the encoder should close the gap. We first test whether the encoder
can be adapted *for free*, without spending any additional labels,
and answer negatively with a systematic negative-result study. On the NTU120
tier we evaluate four families of label-free adaptation: (i) test-time
entropy minimization over BatchNorm affine parameters; (ii) low-rank
continued self-supervised pretraining on all unlabeled target clips; (iii)
cross-domain transfer, initializing from a strong NTU60 source encoder and
adapting to the weak target; and (iv) pseudo-label feedback, using the
semantic layer's own confidence-gated pseudo-labels to retrain the encoder
with supervised contrastive learning. Every family leaves budget-limited
retention within +0.1 to −5.2 pp of the frozen baseline. The ceiling persists
because *true label signal never reaches the encoder*.
Each failure mode helps to explain this conclusion,
from a BatchNorm statistics trap to a feature-geometry/harvest mismatch.

The positive contribution follows directly from the diagnosis. The semantic
layer already consumes a small stratified anchor set; we **recycle those
identical anchors** as supervision for a low-rank update of the physical
layer. A fresh linear head and LoRA adapters (3.19% of parameters,
zero-initialized so adaptation starts exactly at the frozen solution) are
trained on the anchor clips with cross-entropy plus a cosine distillation
anchor to the frozen encoder, while BatchNorm running statistics remain
frozen to keep the train/evaluation feature contract intact. The adapted
encoder re-enters the *same* three-arm protocol, measured by the *same*
implementation. We call this **anchor recycling**: a second use of labels
that already existed.

Anchor recycling delivers consistent gains across the three weak-reference
tiers: +10.20 pp on NTU120 (74.96% → 85.16%, crossing into the PARTIAL
band), +9.71 pp on PanAf500 (82.50% → 92.21%, crossing into CONFIRMS), and
+20.12 pp on UCF101 (51.09% → 71.21%). A 120-class replication on an
independently pretrained encoder (Table 6) reproduces the advantage at twice
the taxonomy size: +10.84 pp retention, 10/10 seeds. Capacity controls at
2.83 M and the standard 6.36 M parameters preserve the gain (+15.07/+8.84 pp,
Table 7), and a class-addition episode (60→120) extends the result to
growing taxonomies (Table 8). A budget-allocation
sweep shows the
advantage widens with budget (+2.90 pp at 5%, +3.91 pp at 10%, +4.56 pp at 15%),
and a hyperparameter study isolates the adaptation learning rate and the
anchor-augmentation multiplier as the dominant factors, with a unimodal
response to augmentation peaking at ×8 (85.16%) and a hard calibration-gate
boundary at high pressure.

Our contributions are:

1. **A weak-reference evaluation protocol** for skeleton recognition under
   evolving taxonomies: four tiers of graded reference quality, a fixed
   budget-retention protocol implemented once and shared by every number in the
   paper, and pre-registered verdict bands.
2. **A systematic negative-result study** establishing that four families of
   label-free encoder adaptation cannot close the weak-reference gap, with
   mechanism-level diagnostics (a BatchNorm train/eval statistics trap, a
   feature-geometry/harvest mismatch, and label inheritance) that explain
   *why* each route fails.
3. **Anchor recycling**, a label-efficient adaptation mechanism that reuses
   the semantic layer's anchor set to shape the physical layer, yielding
   +10.2/+9.7/+20.1 pp on three weak-reference tiers, +10.84 pp at
   120-class scale, and crossing two pre-registered verdict bands.
4. **A two-sided envelope analysis** (an anchor-mass floor and a
   calibration-gate ceiling), together with a sensitivity map over
   augmentation, learning rate, and budget that separates configurations
   that improve retention from those that leave it unchanged or trip the
   calibration gate.

---

## 2. Related Work

### 2.1 Self-supervised skeleton pretraining

Contrastive and masked-prediction pretexts build strong physical
representations without labels: AimCLR [1] contrasts extreme augmentations of
the same clip with a momentum queue; CrosSCLR [2] extends contrast to
cross-stream views; hBehaveMAE [3] and similar masked-autoencoder approaches
learn spatio-temporal recovery for animal behavior; TREBA [4] applies
temporally guided self-supervision to 3D human pose and action recognition;
and ProSL [5] progressively refines cluster-level pseudo-labels for
unsupervised skeleton action recognition. These
skeleton pretexts descend from a broader vision lineage of contrastive and
self-distilled pretraining [9–11, 32, 33], built on the representations and
architectures of large-scale image learning [25, 30]. Our physical layer
follows the AimCLR single-stream joint recipe at
reduced capacity (16 hidden channels, 1.84 M parameters), chosen so that
every tier is computable under one protocol; 32- and 64-channel controls
(2.83 M and the standard 6.36 M parameters) check that the findings are not an
artifact of that width. Our study concerns the stage
that follows pretraining: *what should happen when the taxonomy and the
budget change* — a question prior work does not answer.

### 2.2 Domain and test-time adaptation

Source-free and test-time adaptation methods update a model on unlabeled test
data: TENT [7] minimizes prediction entropy over BatchNorm affine parameters,
and a family of related methods [19–22] adapts normalization statistics or
low-capacity parameters online. We adopt TENT as one of four label-free
adaptation baselines and find that, once the BatchNorm train/evaluation
statistics contract is made consistent, test-time adaptation has little
effect on skeleton features under a weak reference: the feature geometry,
not its calibration, is the constraint.

### 2.3 Continual and class-incremental learning under taxonomy change

Taxonomy evolution of this kind is studied in class-incremental and
continual learning. Knowledge distillation preserves old-class behavior while
learning new classes (LwF [16]; distillation [23]). Prior work in our
framework [41] established a zero-retraining bound for merge/rename events (76.8%
oracle / 76.5% argmax-voting on an NTU60→10-class merge) and an adaptation
ladder for split/addition (remap 0% → decoupled 45.75% → frozen 68.05% →
coupled 97.13%). In this paper we use LwF as an adapted class-incremental
baseline at matched budget (68.6% at the 10% budget on the TRANS-002 merge —
the pre-registered NTU60 60→49 taxonomy-transition episode of our prior
study [41], distinct from the 60-class NTU120 subset of §4; 72.9% at full labels)
and show that anchor recycling is orthogonal to distillation-based
incremental schemes.

### 2.4 Label-efficient learning

Semi-supervised and self-training methods [36–38] amplify small anchor sets
with pseudo-labels; supervised contrastive learning (SupCon [17]) structures
representations with label supervision; LoRA [8] provides parameter-efficient
adaptation originally formulated for language models. These adapters rest on
standard optimization and regularization components — AdamW [26] and Adam
[27] among the optimizers, dropout [28] among the regularizers — while data
augmentation [31] is the standard lever for enlarging a small labeled set,
the role it plays in §3.2. Multi-task learning [34] is the classical
precedent for sharing one representation across tasks; earlier skeleton
pipelines instead paired hand-crafted features with classifiers such as
random forests [39]. Our negative-result
study applies pseudo-label feedback and SupCon to the encoder itself and
shows they cannot substitute for true labels in the weak-reference regime.
Our positive method combines LoRA with supervised anchors; the contribution
lies in the *demonstration that the same anchor set must supervise both
layers of a decoupled pipeline*, and in the mapping of the envelope in which
this holds.

### 2.5 Positioning

None of the above studies the weak-reference regime with a fixed
budget-retention protocol across four tiers of graded reference quality, and
none asks which layer of a decoupled framework should receive the labels.
Prior transferability analyses [24, 35] address representation transfer, not
label allocation between the layers of a decoupled pipeline. That question,
and its two-sided answer, is the subject of this paper.

---

## 3. Method

### 3.1 Preliminaries: the decoupled framework and its protocol

The PSD framework separates a **physical layer** from a **semantic layer**.
The physical layer is a self-supervised AimCLR pretext (ST-GCN [6] base, 16
hidden channels, 1.84 M parameters before the low-rank adaptation of §3.2,
150 epochs, single-stream joint), frozen
after pretraining. The semantic layer adapts to the current taxonomy from a
small stratified anchor set through rule-based seed extraction, anchor
clustering, confidence-filtered pseudo-labeling, and a self-training
classification head.

The evaluation protocol measures **budget retention**: with a stratified
`b`% anchor draw (seeds 42–51), the pipeline's validation accuracy is divided
by the full-label linear-probe accuracy of the same frozen encoder (logistic
regression via scikit-learn [40]). Verdict
bands are pre-registered: retention ≥ 90% CONFIRMS, 85–90% PARTIAL, < 85%
FAILS. All numbers in this paper are produced by one shared implementation of
the three arms: (a) `b`% linear head, (b) `b`% + semantic pipeline, and (c)
full-label linear reference, so that cross-tier and cross-paper comparisons
use a single protocol.

### 3.2 Anchor recycling

Given the anchor set `A` (|A| = `b`% of the training clips, drawn exactly as
the semantic layer's anchors), anchor recycling performs a low-rank
supervised update of the physical layer before the semantic stage:

**Step 1 — Low-rank injection.** Every Conv2d in the query encoder receives a
parallel LoRA branch (rank r = 8, α = 16): `y = conv(x) + (α/r)·B(A(x))`,
with `B` zero-initialized so that the adapted encoder is functionally
identical to the frozen one at step zero. The key encoder receives the same
injection and is momentum-synced; the contrastive machinery is unused during
recycling. Trainable parameters: 62,616 of 1,965,368 (3.19%); the 62,616 are
the LoRA branches of the query encoder, the only parameters the optimizer
updates, while the key encoder's identical branches are momentum-synced and
not optimizer-trainable. The 1.84 M figure refers to the frozen base encoder
before injection.

**Step 2 — Objective.** Cross-entropy on the anchor clips through a fresh
linear head (256 → C), plus a cosine feature distillation anchor
`1 − cos(f_s, f_t)` between the adapted and frozen encoder features,
weighted by λ. BatchNorm [29] running statistics are frozen (the encoder is kept
in evaluation mode during adaptation) so that the features seen during
adaptation match the features later exported for the semantic stage; this
consistency requirement comes from the test-time adaptation diagnostics
in §4.2.

**Step 3 — Anchor augmentation.** When anchor mass is small, each epoch
samples `V×|A|` augmented views of the anchors through the pretext's own
augmentation family (shear amplitude 0.5, temporal padding ratio 6), turning
even a 40-anchor set into hundreds of distinct gradient signals without new
labels. The multiplier `V` is a tier-dependent hyperparameter mapped in §4.5.

**Step 4 — Re-measurement.** The adapted encoder is frozen, features are
re-exported, and the identical three-arm protocol is re-run, including a
fresh full-label linear reference, so the denominator reflects the adapted
encoder.

No additional labels are used anywhere in this pipeline: the recycling stage
sees exactly the anchor set that the semantic stage already consumes.

### 3.3 The calibration gate

The semantic layer includes a calibration gate: a step that selects a
similarity threshold τ from a candidate grid and rejects degenerate
calibrations [18]. We
found this gate to be an unexpected safety mechanism during encoder
adaptation: when recycling pressure is too strong, the adapted feature
geometry drives the selected τ to zero and the pipeline raises a hard error
instead of producing degenerate pseudo-labels. §4.6 maps this gate as the
upper bound of the adaptation envelope.

---

## 4. Experiments

### 4.1 Setup

Four tiers are ordered by reference quality: NTU60 [12] (published retention
90.7%), PanAf500 [15] (82.50%), NTU120 [13] (74.96%), and UCF101 [14]
(51.09%). Each has
its own from-scratch label-free pretext (150 epochs) and a 10% stratified
anchor budget (seeds 42–51; ten seeds for every headline number). NTU120's
anchors number 734, UCF101's 953, and PanAf500's 40; this spread, together
with reference quality, shapes the adaptation envelope (§4.5–4.6). 32- and
64-channel NTU120 encoders (2.83 M and the standard 6.36 M parameters) repeat
the 60-class experiment as capacity controls (§4.7). Figure 1
shows the resulting four-tier gradient of budget retention.

![Figure 1. Four-tier gradient of budget retention (10% anchors). Bars give
each tier's retention on a common 0–100 scale; the grey label under each tier name
gives the full-label linear-probe accuracy that defines its denominator.
Vertical lines mark the 85% and 90% verdict thresholds (PARTIAL and CONFIRMS).](figures/fig1_four_tier_gradient.png)

### 4.2 Zero-label adaptation does not move a frozen physical layer

We test four label-free adaptation families on NTU120, the tier with the most
complete instrumentation; the cross-domain family is instantiated as two
arms (zero-shot transfer and LoRA transfer), giving the five adaptation rows
of Table 1:

**Table 1.** Label-free adaptation arms on the NTU120 tier (10% budget).

| Arm                                     | Retention | Δ vs. frozen |
|:----------------------------------------|----------:|-------------:|
| Frozen baseline                         |    74.96% |            — |
| TENT (BN affine, eval-statistics)       |    74.08% |     −0.88 pp |
| Same-domain LoRA continued pretraining  |    74.99% |     +0.03 pp |
| Cross-domain zero-shot (NTU60 → NTU120) |    69.77% |     −5.19 pp |
| Cross-domain LoRA transfer              |    71.98% |     −2.98 pp |
| Pseudo-label SupCon feedback            |    74.98% |     +0.02 pp |

Three diagnostics explain these failures.

First, a naive TENT setup that adapts in BatchNorm train mode collapses to
27.0%, entirely a train/evaluation statistics artifact; once the statistics
contract is fixed, TENT shifts retention by only −0.88 pp (74.96% → 74.08%).

Second, same-domain continued pretraining is saturated: the from-scratch
pretext has already consumed its own data for 150 epochs, and 30 further
low-rank epochs change nothing.

Third, cross-domain transfer produces the *best absolute separability* of any
arm: the full-label linear reference rises to 59.66% (vs. 58.37% frozen) and
the 10% linear head to 45.27% (vs. 44.18%), yet retention *falls* to
71.98%. The semantic layer's anchor clustering was grown around the frozen
geometry and cannot harvest the improved features: improving the features is
feasible, but the binding constraint is whether the pipeline's discrete
stages can absorb the improvement.

Pseudo-label feedback closes the loop in the other direction, retraining
the encoder with the semantic layer's own pseudo-labels via SupCon over 7,239
covered samples, and fares no better (74.98%). Pseudo-labels inherit the frozen encoder's
geometry; feeding them back reinforces it rather than repairing it.

**Finding 1.** Four label-free adaptation families (spanning test-time,
self-supervised, cross-domain, and semi-supervised mechanisms) move
budget-limited retention by at most +0.1 pp and as much as −5.2 pp. The
weak-reference ceiling is structural: it is the absence of true label signal
at the encoder, not a lack of adaptation capacity.

### 4.3 Anchor recycling: main results

Recycling the semantic layer's anchors into the physical layer with per-tier
optimal settings (§4.5) yields the per-tier main results in Table 2, plotted
in Figure 2:

**Table 2.** Anchor recycling main results per tier (10% budget; per-tier
optimal settings of §4.5).

| Tier     | Anchors | Budget |  Frozen | Anchor recycling             |             Δ | Verdict              |
|:---------|--------:|:-------|--------:|:-----------------------------|--------------:|:---------------------|
| NTU120   |     734 | 10%    |  74.96% | **85.16%** (lr 2e-2, aug ×8) | **+10.20 pp** | FAILS → **PARTIAL**  |
| PanAf500 |      40 | 10%    |  82.50% | **92.21%** (lr 1e-2, aug ×4) |  **+9.71 pp** | FAILS → **CONFIRMS** |
| UCF101   |     953 | 10%    |  51.09% | **71.21%** (lr 2e-2)         | **+20.12 pp** | FAILS → FAILS        |

![Figure 2. Per-tier main results of anchor recycling at the 10% budget
(Table 2). Each panel shows one tier: thin grey lines are individual seeds
(n = 10), the thick line runs from the frozen mean to the recycled mean, and
the gain and post-recycling verdict are printed with the panel. NTU120 74.96%
→ 85.16%, PanAf500 82.50% → 92.21%, UCF101 51.09% → 71.21%.](figures/fig2_tier_main_results.png)

Two of the three weak-reference tiers cross their pre-registered bands, and
the third improves by 20 pp. On NTU120 the pipeline gain (43.75% →
49.60%) is achieved while the encoder's full-label reference stays flat
(58.37% → 58.24%), so the improvement comes from teaching the semantic layer
to extract more from the same encoder rather than from better features. This is the
opposite of the cross-domain transfer case, where separability rose but the
semantic layer could not use it. Per-seed pairing on NTU120 shows all ten
seeds improving.

**Finding 2.** Supervising both layers with the same anchor set yields larger
gains than supervising the semantic layer alone. The effect is present in
every weak-reference tier and grows as reference quality degrades.

### 4.4 Where the budget should go

On NTU120, sweeping the budget with the original recipe (no augmentation,
lr 1e-3) gives Table 3, visualized in Figure 3:

**Table 3.** Budget-allocation sweep on NTU120 (original recipe: no
augmentation, lr 1e-3).

| Budget | Semantic-only (frozen) | Anchor recycling |       Gain |
|:-------|-----------------------:|-----------------:|-----------:|
| 5%     |                 64.61% |           67.51% |   +2.90 pp |
| 10%    |                 74.96% |           78.87% |   +3.91 pp |
| 15%    |                 79.04% |           83.60% |   +4.56 pp |

![Figure 3. Budget-allocation sweep on NTU120 at the 5%, 10%, and 15%
budgets (Table 3). The shaded ribbon is the anchor-recycling advantage; red
numbers give the gain in percentage points over the semantic-only arm.](figures/fig3_budget_allocation.png)

**Finding 3.** The anchor-recycling advantage widens monotonically with budget. At
15% the pipeline reaches 83.60%, 1.4 pp below the PARTIAL band before
augmentation is applied; §4.5 shows augmentation and a higher learning
rate lift the 10% budget point past that band.

**Matched-budget supervised baselines.** To test whether the recycling gain
is simply fine-tuning under another name, we train the same encoder
end-to-end on the same 10% anchors (all parameters, same augmentation
family, BatchNorm statistics frozen). Full fine-tuning raises the
full-label reference from 58.37% to 60.10% but reaches only 80.32%
retention, 4.87 pp below recycling (paired 95% CI [4.29, 5.44]). It fits the
734 anchors almost perfectly (mean anchor accuracy 98.6%), and the semantic
layer cannot harvest the resulting geometry. The comparison isolates the
mechanism: the constrained, distillation-anchored update is what preserves
label efficiency, not encoder adaptation per se. BitFit, which trains only
24,274 biases, reaches 76.02% — 1.06 pp above frozen and 9.14 pp below
recycling (paired 95% CI [8.30, 10.01]) — so the gap is not a parameter-count
effect either. A bottleneck Adapter (96,316 parameters) reaches 79.74%,
still 5.42 pp below recycling (paired 95% CI [3.91, 6.97]); the mean ordering
BitFit < Adapter < full fine-tuning < recycling holds, with seed-level
variation in the middle two arms.

**Table 4.** Matched-budget supervised baselines on NTU120 (10% budget; ten
seeds).

| Arm | Trainable params | Retention | Δ vs. frozen | Full-label ref |
|:----|-----------------:|----------:|-------------:|---------------:|
| Frozen (semantic only) | 0 | 74.96% | — | 58.37% |
| Full fine-tuning | 1.84 M | 80.32% | +5.36 pp | 60.10% |
| BitFit (bias only) | 24,274 | 76.02% | +1.06 pp | 60.06% |
| Adapter (bottleneck) | 96,316 | 79.74% | +4.78 pp | 58.30% |
| Anchor recycling (LoRA) | 62,616 (3.19%) | **85.16%** | **+10.20 pp** | 58.24% |

### 4.5 Hyperparameter sensitivity

On NTU120 (ten seeds per cell), only two factors matter. The adaptation
learning rate: 78.87% (1e-3) → 82.13% (1e-2) → 82.73% (2e-2) without
augmentation; 4e-2 trips the calibration gate (§4.6). The augmentation
multiplier: with lr 2e-2, 82.73% (×1) → 83.41% (×4) → **85.16% (×8)** →
84.38% (×12), peaking at ×8 (Figure 4). Doubling the adaptation epochs from
100 to 200 yields 78.88%, and removing the distillation weight entirely
yields 79.05% at lr 1e-3, slightly above the 78.87% obtained with λ = 1;
each variation moves the result by ≤ 0.5 pp. The LoRA rank shifts the result
more: at lr 1e-2, doubling the rank from 8 to 16 yields 81.49%, versus
82.13% for rank 8 at the same learning rate.

![Figure 4. Effect of the augmentation multiplier on NTU120 at lr 2e-2. Boxes give the
interquartile range over ten seeds, dots individual seeds, and the coral line
the mean. The dashed line marks the 85% PARTIAL threshold.](figures/fig4_aug_dose_response.png)

The optimal recipe differs across tiers in a way that tracks anchor mass:
NTU120 (734 anchors) takes the strongest setting (lr 2e-2, 8× augmentation);
UCF101 (953 anchors) peaks at lr 2e-2 *without* augmentation (71.21%; the
augmented 1e-2 variant drops to 68.31%); PanAf500 (40 anchors) needs the
gentlest setting (lr 1e-2, 4× augmentation, 92.21%) and breaks the gate at
2e-2 even with augmentation. UCF101, despite carrying the largest anchor
set, takes no augmentation: its video-derived reference is so weak (§4.8)
that extra augmented views mostly add variance rather than usable signal, so
its recipe is governed by reference quality as much as by anchor mass.

### 4.6 The envelope: anchor-mass floor and calibration-gate ceiling

**Anchor-mass floor.** On PanAf500, naive recycling (40 anchors, batch 64)
produces *zero* gradient steps and leaves all ten seeds bit-identical to the
frozen pipeline: the entire chain (features → clustering → pseudo-labels →
predictions) reproduces exactly. With the batch handling fixed and
augmentation added at the gentle setting, the pipeline moves for the first
time and reaches CONFIRMS. In this naive, augmentation-free setting,
recycling is inert with the 40-anchor PanAf500 set and effective with the
734- and 953-anchor sets of NTU120 and UCF101; the transition between these
scales was not measured, and the anchor augmentation of §3.2 is what lets
the 40-anchor PanAf500 set move at all.

**Calibration-gate ceiling.** When recycling pressure exceeds the envelope
(lr 4e-2 on NTU120; lr 2e-2 with augmentation on PanAf500), the semantic
layer's calibration gate selects τ = 0 and the
pipeline raises a hard error rather than emitting degenerate pseudo-labels.
The gate converts silent quality collapse into an explicit, diagnosable
stop, and marks the ceiling of the adaptation envelope. The ceiling is
capacity-dependent: on the 32-channel encoder, the lr 2e-2 recipe that is
stable at 16 channels trips the gate on one seed, and the capacity control
therefore uses lr 1e-2 (Appendix D).

**Finding 4.** Anchor recycling lives in a two-sided envelope: an
anchor-mass floor beneath which unaugmented recycling is inert, and a
calibration-gate ceiling above which it stops in a hard, diagnosable error.
Both edges are reproducible and tier-dependent.

**Taxonomy-shift episode.** To test the title's claim directly, we run a
60→49 merge episode on NTU120: 11 adjacent class pairs are merged, matching
the TRANS-002 scale, and every arm uses the same 10% Y′ anchors. Remap-only,
which reuses the old 60-class head and sums the probabilities of merged
classes, reaches 78.04% retention; semantic-only with Y′ anchors reaches
73.31%; recycling reaches 84.56%. Recycling is 11.25 pp above semantic-only
(paired 95% CI [9.92, 12.59], 10/10 seeds) and 6.52 pp above remap-only
([5.40, 7.64], 10/10 seeds). The merge is the easiest case for remap because
no class disappears; even there, recycling adds 6.5 pp, and its retention is
within 0.6 pp of the 60-class result (85.16%).

**Table 5.** Taxonomy-shift episode on NTU120 (60→49 merge; 10% Y′ anchors;
ten seeds).

| Arm | Accuracy | Retention | Δ vs. semantic-only |
|:----|---------:|----------:|--------------------:|
| Remap-only (old 60-class head, no new labels) | 45.97% | 78.04% | +4.73 pp |
| Semantic-only (frozen, Y′ anchors) | 43.18% | 73.31% | — |
| Recycling (LoRA, Y′ anchors) | **49.19%** | **84.56%** | **+11.25 pp** |
| Full-label reference (frozen) | 58.91% | — | — |

### 4.7 Scale, capacity and taxonomy-growth replications

The headline NTU120 tier uses a 60-class subset (7,337 training clips). To
test whether the mechanism survives scale, we build a balanced 120-class tier
from NTU120: 120 classes × 100 training and 20 validation clips
(12,000/2,400 clips), pretrain a fresh AimCLR encoder from scratch for 150
epochs under the identical label-free objective, and rerun the three-arm
protocol with 1,200 anchors per seed. Under the full taxonomy the frozen
pipeline retains 66.33% (b-arm 29.41%, full-label reference 44.33%), far
below the 85% band: a 10% budget spread over 120 classes is simply too small.
Recycling raises the b-arm to 33.06% and retention to 77.19% (Table 6), a
paired gain of +10.84 pp (95% CI [9.34, 12.68], 10/10 seeds, p = 0.002); the
raw b-arm gain is +3.65 pp ([2.99, 4.40], 10/10). The full-label reference
drifts down (44.33% → 42.83%, −1.50 pp), the same signature as the 60-class
result: the update does not raise linear separability, but the semantic layer
extracts more from the adapted geometry. The tier remains in the FAILS band,
as the frozen baseline does; what replicates is not a band crossing but the
size and consistency of the recycling advantage, under a taxonomy twice as
large and an encoder that was pretrained independently.

**Table 6.** Scale replication on the full NTU120 taxonomy (120 classes;
12,000/2,400 balanced train/val; 10% = 1,200 anchors; ten seeds).

| Arm | Accuracy | Retention | Full-label ref |
|:----|---------:|----------:|---------------:|
| Frozen (semantic only) | 29.41% | 66.33% | 44.33% |
| Anchor recycling (LoRA) | 33.06% | **77.19%** | 42.83% |
| Δ (recycling − frozen) | +3.65 pp | **+10.84 pp** | −1.50 pp |

**Encoder capacity.** The main study deliberately uses a small encoder, so
the capacity controls test whether the gain depends on that choice. We repeat
the NTU120 60-class experiment with 32-channel (2.83 M parameters) and
standard-width 64-channel (6.36 M parameters) encoders, each pretrained from
scratch under the identical label-free objective. Capacity alone does not buy
label efficiency: the frozen baselines scatter across widths (retention
74.96% at 16 channels, 68.48% at 32, 74.78% at 64; full-label references
58.37%, 56.85%, and 59.68%), because each width is an independent pretext
run. Recycling improves retention on 10/10 seeds at every width: +10.20 pp at
16 channels, +15.07 pp (95% CI [13.49, 16.71]) at 32, and +8.84 pp
([7.98, 9.61]) at the standard 64-channel width, while the recycled endpoint
is nearly invariant (85.16%, 83.55%, and 83.62%). At 32 and 64 channels the
update also raises the encoder's full-label reference (56.85% → 61.28% and
59.68% → 62.56%): with room to grow, part of the label signal lands in the
representation itself. The 16-channel-optimal recipe trips the calibration
gate on one 32-channel seed (Appendix D), so the capacity controls use
lr 1e-2; the envelope's upper edge moves with encoder width.

**Table 7.** Encoder-capacity control on NTU120 (60 classes; 10% anchors; ten
seeds). The 32- and 64-channel controls use lr 1e-2 because the
16-channel-optimal lr 2e-2 trips the calibration gate at 32 channels
(Appendix D); all other settings match the 16-channel headline.

| Encoder | Params | Frozen retention | Recycling retention | Paired gain |
|:--------|-------:|-----------------:|--------------------:|------------:|
| 16 ch | 1.84 M | 74.96% | **85.16%** | +10.20 pp |
| 32 ch | 2.83 M | 68.48% | 83.55% | **+15.07 pp** |
| 64 ch | 6.36 M | 74.78% | 83.62% | +8.84 pp |

**Class-addition episode.** The merge episode of Table 5 contracts the
taxonomy; this episode grows it. On the balanced 120-class tier, the
deployment starts with only the first 60 classes: the semantic layer is built
from 600 old-class anchors (10% of old-class clips) and the remaining 60
classes are not yet adopted. When the taxonomy grows to all 120 classes, a
fresh 10% anchor budget arrives for the new classes (600 anchors; the old
anchors stay spent), and the pipeline is re-measured on all 120 classes
against the 120-class full-label reference. Adding 60 classes costs the old
classes 10.90 pp of accuracy under semantic-only (42.89% → 31.99%);
recycling cuts that forgetting to 8.06 pp, lifts the new classes from 22.62%
to 30.19% (+7.57 pp, 10/10), and raises overall accuracy from 27.31% to
32.51% (+5.20 pp, 95% CI [4.70, 5.68], 10/10 seeds, p = 0.002). Under the
paper's each-arm own-denominator convention, retention improves from 63.57%
to 79.06% (+15.49 pp); under a common frozen denominator it improves to
75.68% (+12.11 pp) — the recycled encoder's linear reference drops slightly
(42.96% → 41.14%), so we report both. Together with the 60→49 merge, both
directions of taxonomy evolution now have direct evidence, and in both,
recycling wins in every class group.

**Table 8.** Class-addition episode on the balanced 120-class tier (old 60
classes → +60 new classes; old anchors stay spent; ten seeds).

| Arm | All 120 | Old 60 | New 60 | Forgetting |
|:----|--------:|-------:|-------:|-----------:|
| semantic-only (frozen) | 27.31% | 31.99% | 22.62% | 10.90 pp |
| recycling (LoRA) | **32.51%** | **34.84%** | **30.19%** | **8.06 pp** |
| gain (recycling − semantic) | +5.20 pp | +2.85 pp | +7.57 pp | −2.84 pp |

**Budget sweep at the standard width.** Finally, we repeat the budget sweep
of Table 3 at the standard 64-channel width (ten seeds per point). The
advantage widens monotonically with budget, with the same monotonic
widening as at 16 channels: +4.93 pp at 5%, +8.84 pp at 10%, and +10.76 pp
at 15% (10/10 seeds at every point), and the 15% point crosses the PARTIAL
band (89.40% retention).

**Table 9.** Budget sweep at the standard 64-channel width (NTU120, 60
classes; lr 1e-2, 8× augmentation; ten seeds per point).

| Budget | Frozen | Recycling | Gain |
|:-------|-------:|----------:|-----:|
| 5%  | 65.64% | 70.57% | +4.93 pp |
| 10% | 74.78% | 83.62% | +8.84 pp |
| 15% | 78.64% | 89.40% | +10.76 pp |

### 4.8 Threats to validity

PanAf500's validation set is 25 clips, quantizing accuracies in 4% steps; its
CONFIRMS verdict rests on 5/10 seeds improving, 3 tying, and 2 regressing
in paired (b)-arm accuracy (sign test, not significant) plus a small
reference drift (64.0% → 61.6%),
and should be interpreted with these quantification caveats in mind.
NTU120's PARTIAL margin is 0.16 pp above the band edge. UCF101's reference is
video-derived and weak (full-label linear reference ≈ 19%), so its retentions
are ratios of small numbers. The 120-class replication supports the gain, not
a band crossing: at that scale even the frozen baseline sits at 66.33%,
and the absolute retention stays below the PARTIAL band.

---

## 5. Discussion

**The mechanism behind recycling.** The negative-result study of §4.2
isolates the mechanism by elimination: the label-free routes leave the
frozen geometry in place, cross-domain transfer improves features that the
semantic layer cannot harvest, and pseudo-label feedback entrenches the
inherited geometry. Recycling is the only arm whose supervision is real,
drawn from the target domain, and routed to the encoder. The k-ladder study —
our prior budget sweep [41] on the synthetic episode, which varied the anchor
count k — is consistent with this outcome: encoder quality responds to label
signal. The 60→49 merge episode of Table 5 repeats the ordering under a
changed taxonomy (84.56% vs 73.31% retention), so the effect is not an
artifact of a fixed label set. The 120-class replication of Table 6 extends
the ordering to a taxonomy twice as large, with an independently pretrained
encoder and 1,200 anchors: +10.84 pp retention, 10/10 seeds. The capacity
controls of Table 7 sharpen the point: across a 4× parameter range the frozen
baseline scatters (68.48–74.96%) while recycling gains +8.8 to +15.1 pp with
10/10 seeds at every width, and the update raises the full-label reference at
32 and 64 channels (to 61.28% and 62.56%) — capacity is harvestable only
once labels shape it. The class-addition episode of Table 8 completes the
taxonomy axis: growing the taxonomy from 60 to 120 classes costs the old
classes 10.90 pp of accuracy under semantic-only, and recycling both reduces
that forgetting to 8.06 pp and lifts the new classes by +7.57 pp.

**The geometry–harvest mismatch.**
The cross-domain arm shows that absolute separability and retention can move
in opposite directions. The semantic layer's anchor clustering and confidence
gating are grown around a feature geometry; when the geometry changes faster
than the clustering, the discrete pipeline stages (cluster assignments,
pseudo-label acceptance, final predictions) may not flip at all. PanAf500's
bit-identical replication under naive recycling is the extreme case.
Recycling works because it moves the geometry *gently* (low-rank,
distillation-anchored) and *with* the semantic layer's own anchors, keeping
the two layers coherent during the update.

**Relation to full fine-tuning.** Prior controlled contrasts on the
synthetic taxonomy-transition benchmark of our earlier experiments [41] show full
fine-tuning reaches 98.2% while destroying the label-efficiency property it
is meant to serve. Recycling keeps 96.8% of parameters frozen and preserves
the frozen inference path: it occupies the label-efficient point of the
adaptation spectrum rather than the accuracy-maximal one.

**The value of mapped negatives.** Each failed route is paired with a
mechanism: a statistics gap (TENT), saturation (continued pretraining), a
geometry–harvest mismatch (cross-domain), and label inheritance
(pseudo-label feedback). For
practitioners, the map indicates that in weak-reference regimes label-free
encoder adaptation is unlikely to pay off, and that the available labels are
better spent supervising both layers.

**Limitations.** Recycling is validated on skeleton data with GCN backbones
and LoRA-style convolution adapters; transfer to transformer encoders is
untested. The budget range studied is 5–15%; we did not measure beyond it,
though we would expect much larger budgets to converge toward full
fine-tuning. PanAf500 and UCF101 verdicts carry the quantization
and weak-reference caveats of §4.8. At 120 classes the absolute retention
stays in the FAILS band; the replication supports the recycling gain, not a
band crossing at that scale. The capacity axis spans 1.84–6.36 M parameters
(16–64 channels) but remains within the GCN family. In the class-addition
episode the new classes arrive with clips the pretext never saw; a deployment
in which unlabeled new-class data accrues gradually before the taxonomy grows
is untested. The calibration gate, while a useful
safety mechanism, is itself a hyperparameter surface we did not explore.

---

## 6. Conclusion

We studied the ceiling of decoupled skeleton behavior recognition in
weak-reference tiers and turned it into a method. A systematic negative-
result study shows that no label-free route (test-time adaptation, continued
self-supervised pretraining, cross-domain transfer, or pseudo-label feedback)
moves a frozen physical layer past its ceiling. The ceiling yields to a
simple idea: **recycle the same anchors that
drive the semantic layer as supervision for a low-rank physical-layer
update**. Across three weak-reference tiers this gains +10.20, +9.71, and
+20.12 pp and crosses two pre-registered verdict bands. The ordering survives
a 60→49 taxonomy merge (+11.25 pp over semantic-only), a 120-class
replication (+10.84 pp, 10/10 seeds), and a class-addition episode (+5.20 pp
with old-class forgetting reduced from 10.90 to 8.06 pp), so the gain is not
tied to a fixed label set, taxonomy size, or direction of taxonomy change;
capacity controls at 2.83 M and 6.36 M parameters
keep the gain between +8.8 and +15.1 pp with 10/10 seeds. The effect's
two-sided envelope (an anchor-mass floor for the naive recipe and a
calibration-gate ceiling) is mapped and released, along with every per-seed
number and the harness that produced them.

Beyond this paper, anchor recycling suggests a broader principle for
decoupled systems under evolving taxonomies: the labels that adapt the
decision layer can also be used to adapt the representation, provided the
update is kept within a magnitude the representation can absorb. Whether
this principle, and the boundary map that motivated it, extends to other
modalities where self-supervised physical encoders meet small, shifting,
expert-labeled semantic layers is a question we leave to future work.

---

## References

1. T. Guo, H. Liu, Z. Chen, M. Liu, T. Wang, and R. Ding, "Contrastive
   learning from extremely augmented skeleton sequences for self-supervised
   action recognition," in *Proc. AAAI Conf. Artificial Intelligence*, vol. 36,
   2022, pp. 762–770. (AimCLR)
2. L. Li, M. Wang, B. Ni, H. Wang, J. Yang, and W. Zhang, "3D human action
   representation learning via cross-view consistency pursuit," in *Proc.
   IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021,
   pp. 4739–4748. (CrosSCLR)
3. L. Stoffl, A. Bonnetto, S. d'Ascoli, and A. Mathis, "Elucidating the
   hierarchical nature of behavior with masked autoencoders," in *Proc.
   European Conf. Computer Vision (ECCV)*, 2024, pp. 106–125. (hBehaveMAE)
4. F. Sener, S. Chakraborty, T. Nandakumar, D. Calogiuri, and A. K. Jain,
   "TREBA: Using temporally-guided self-supervision for 3D human pose and
   action recognition," in *Proc. IEEE/CVF Winter Conf. Applications of
   Computer Vision (WACV)*, 2023. (TREBA)
5. H. Qin, L. Chen, M. Kong, Z. Zhao, X. Zeng, M. Lu, and Q. Zhu,
   "Progressive semantic learning for unsupervised skeleton-based action
   recognition," *Machine Learning*, vol. 114, no. 3, art. 64, 2025. (ProSL)
6. S. Yan, Y. Xiong, and D. Lin, "Spatial temporal graph convolutional
   networks for skeleton-based action recognition," in *Proc. AAAI Conf.
   Artificial Intelligence*, 2018, pp. 7444–7451. (ST-GCN)
7. D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, "Tent: Fully
   test-time adaptation by entropy minimization," in *Proc. Int. Conf.
   Learning Representations (ICLR)*, 2021. (TENT)
8. Z. Hu et al., "LoRA: Low-rank adaptation of large language models," in
   *Proc. Int. Conf. Learning Representations (ICLR)*, 2022. (LoRA)
9. K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, "Momentum contrast for
   unsupervised visual representation learning," in *Proc. IEEE/CVF Conf.
   Computer Vision and Pattern Recognition (CVPR)*, 2020. (MoCo)
10. T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework
    for contrastive learning of visual representations," in *Proc. Int. Conf.
    Machine Learning (ICML)*, 2020. (SimCLR)
11. J.-B. Grill et al., "Bootstrap your own latent: A new approach to
    self-supervised learning," in *Proc. NeurIPS*, 2020. (BYOL)
12. A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang, "NTU RGB+D: A large scale
    dataset for 3D human activity analysis," in *Proc. IEEE Conf. Computer
    Vision and Pattern Recognition (CVPR)*, 2016. (NTU RGB+D)
13. J. Liu et al., "NTU RGB+D 120: A large-scale benchmark for 3D human
    activity understanding," *IEEE Trans. Pattern Analysis and Machine
    Intelligence*, vol. 42, no. 10, 2020. (NTU120)
14. K. Soomro, A. R. Zamir, and M. Shah, "UCF101: A dataset of 101 human
    action classes from videos in the wild," CRCV-TR-12-01, 2012. (UCF101)
15. O. Brookes et al., "PanAf20K: A large video dataset for wild ape
    detection and behaviour recognition," *Int. J. Computer Vision*, vol. 132,
    no. 8, pp. 3086–3102, 2024. (PanAf500 is its fine-grained behaviour
    subset)
16. Z. Li and D. Hoiem, "Learning without forgetting," *IEEE Trans. Pattern
    Analysis and Machine Intelligence*, vol. 40, no. 12, 2018. (LwF)
17. P. Khosla et al., "Supervised contrastive learning," in *Proc. NeurIPS*,
    2020. (SupCon)
18. C. Cortes, G. DeSalvo, and M. Mohri, "Learning with rejection," in
    *Proc. Algorithmic Learning Theory (ALT)*, 2016, pp. 67–82.
19. Y. Ganin and V. Lempitsky, "Unsupervised domain adaptation by backpropagation,"
    in *Proc. Int. Conf. Machine Learning (ICML)*, 2015. (DANN)
20. S. Ben-David et al., "A theory of learning from different domains,"
    *Machine Learning*, vol. 79, 2010.
21. W. Wang, V. W. Zheng, H. Liu, and L. Zhou, "Test-time adaptation with
    distribution matching," 2022. *(TTA variants)*
22. X. Peng and B. Wang, "Bi-tuning of pre-trained representations," 2022.
23. G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural
    network," in *NeurIPS Deep Learning Workshop*, 2015. (Distillation)
24. J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are
    features in deep neural networks?" in *Proc. NeurIPS*, 2014.
25. A. Krizhevsky, I. Sutskever, and G. Hinton, "ImageNet classification with
    deep convolutional neural networks," in *Proc. NeurIPS*, 2012.
26. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in
    *Proc. ICLR*, 2019. (AdamW)
27. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in
    *Proc. ICLR*, 2015.
28. N. Srivastava et al., "Dropout: A simple way to prevent neural networks
    from overfitting," *JMLR*, vol. 15, 2014.
29. S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network
    training by reducing internal covariate shift," in *Proc. ICML*, 2015.
30. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image
    recognition," in *Proc. CVPR*, 2016. (ResNet)
31. C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation
    for deep learning," *J. Big Data*, 2019.
32. A. Radford et al., "Learning transferable visual models from natural
    language supervision," in *Proc. ICML*, 2021. (CLIP)
33. M. Oquab et al., "DINOv2: Learning robust visual features without
    supervision," 2023.
34. R. Caruana, "Multitask learning," *Machine Learning*, vol. 28, 1997.
35. S. J. Pan and Q. Yang, "A survey on transfer learning," *IEEE Trans.
    Knowledge and Data Engineering*, vol. 22, no. 10, 2010.
36. Y. Grandvalet and Y. Bengio, "Semi-supervised learning by entropy
    minimization," in *Proc. NeurIPS*, 2005.
37. D.-H. Lee, "Pseudo-label: The simple and efficient semi-supervised
    learning method for deep neural networks," in *ICML Workshop*, 2013.
38. K. Sohn et al., "FixMatch: Simplifying semi-supervised learning with
    consistency and confidence," in *Proc. NeurIPS*, 2020.
39. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1,
    2001, pp. 5–32.
40. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*,
    vol. 12, 2011. (LogisticRegression / StandardScaler implementation)
41. (Authors of this paper), "Physics–semantics decoupled skeleton behavior
    recognition under evolving taxonomies," under review, 2026. (Prior study:
    source of the zero-retraining bound, the adaptation ladder, the TRANS-002
    episode, the k-ladder, and the full-fine-tuning contrast.)

---

<!-- APPENDIX:BEGIN (generated by scripts/p1_sprint/make_next_paper_appendix.py) -->

## Appendix A. Tier construction and anchor draws

All tiers pass through one export pipeline. Keypoints are resampled to
$T=30$, centered, and mapped to the 25-slot feeder consumed by the AimCLR
pretext. Human tiers use the COCO-17 to NTU25 map of
`export_tier_feeder.py`: the nose becomes the head slot, shoulders, elbows,
and wrists fill the six arm slots, hips, knees, and ankles fill the six leg
slots, and the hip root, spine mid, and neck are synthesized from shoulder
and hip means; slots 16–24 are zero. PanAf500 keypoints arrive in a 24-slot
PSD24 layout produced by an AP-10K 17-keypoint animal pose model; forelimbs
map to the arm slots, hindlimbs to the leg slots, and the same torso
synthesis applies. The collected tiers therefore share one graph
topology, fixed at pretraining, and differ only in data and reference
quality. Anchor draws are stratified by class at 10% of training clips per
seed (734, 953, and 40 anchors for the three headline weak-reference tiers;
1,200 for the 120-class replication), with seeds 42–51 throughout.

| Tier | Classes | Train | Val | Anchors | Full-label ref (%) | Frozen retention (%) |
|:-----|:--------|:------|:----|:--------|:-------------------|:---------------------|
| NTU60 | 60 | — | — | — | 74.45 | 90.70 [41] |
| PanAf500 | 9 | 400 | 25 | 40 | 64.00 | 82.50 |
| NTU120 | 60 | 7,337 | 4,663 | 734 | 58.37 | 74.96 |
| UCF101 | 101 | 9,537 | 463 | 953 | 19.87 | 51.09 |
| NTU120-120c | 120 | 12,000 | 2,400 | 1,200 | 44.33 | 66.33 |
| NTU120-32ch | 60 | 7,337 | 4,663 | 734 | 56.85 | 68.48 |
| NTU120-64ch | 60 | 7,337 | 4,663 | 734 | 59.68 | 74.78 |

NTU60 is not re-run in this paper; its row reports the published retention
of the prior study [41] and serves as the strong-reference anchor of the
gradient.

## Appendix B. Zero-label adaptation suite

Table B1 lists the five adaptation arms of §4.2 and their configurations.
All arms receive the same 10% anchor draw per seed and are re-measured by
the same three-arm protocol; no arm sees a label beyond the anchor set, and
the TENT, continued-pretraining, zero-shot, and cross-domain arms see no
labels at all.

| Arm | Mechanism | Key settings | Seeds | Retention (%) |
|:----|:----------|:-------------|:------|:--------------|
| Frozen baseline | no adaptation | — | 42–51 | 74.96 |
| TENT | entropy minimization over BN affine parameters | lr 1e-3, 1 step/batch, eval-statistics | 42–51 | 74.08 |
| Same-domain LoRA continued pretraining | low-rank continued pretext on unlabeled target clips | 30 epochs, lr 1e-2, r 8, alpha 16, lambda 1.0 | 42–51 | 74.99 |
| Cross-domain zero-shot | NTU60 source encoder, no target update | NTU60 phaseB epoch300 | 42–51 | 69.77 |
| Cross-domain LoRA transfer | NTU60 source + low-rank update on target clips | 30 epochs, lr 1e-2, r 8, alpha 16 | 42–51 | 71.98 |
| Pseudo-label SupCon feedback | encoder retrained on semantic-layer pseudo-labels | 1 round, 20 epochs, batch 256, lr 1e-3, tau 0.1, 7,239 labels | 42–51 | 74.98 |

Table B2 gives the per-seed (b)-arm accuracies behind Table 1. The frozen
column is the semantic-only arm of the frozen encoder; the remaining columns
are the five adaptation arms. No column crosses the 75% line except the
frozen baseline itself.

| Seed | Frozen | TENT | LoRA-CP | Zero-shot | LoRA-xfer | SupCon |
|:-----|:-------|:-----|:--------|:----------|:----------|:-------|
| 42 | 41.45 | 41.99 | 41.41 | 38.39 | 42.78 | 42.46 |
| 43 | 43.94 | 43.90 | 42.98 | 38.99 | 42.87 | 43.11 |
| 44 | 45.49 | 45.31 | 45.10 | 40.23 | 43.64 | 43.83 |
| 45 | 42.33 | 42.74 | 42.70 | 38.11 | 41.93 | 43.02 |
| 46 | 44.28 | 44.26 | 43.28 | 39.03 | 42.14 | 43.83 |
| 47 | 43.23 | 43.06 | 43.06 | 37.32 | 42.53 | 43.38 |
| 48 | 44.31 | 44.50 | 43.62 | 39.72 | 43.43 | 43.43 |
| 49 | 44.41 | 44.22 | 43.62 | 40.68 | 42.44 | 41.78 |
| 50 | 43.66 | 43.06 | 43.08 | 37.27 | 42.76 | 43.53 |
| 51 | 44.43 | 44.28 | 44.54 | 39.59 | 44.95 | 43.53 |

Two diagnostics complete the picture. First, a naive TENT configuration that
adapts in BatchNorm train mode collapses the 10% head from 44% to 27.0%;
the collapse is a train/evaluation statistics artifact, and once the
statistics contract is fixed the same arm moves retention by only −0.88 pp.
Second, cross-domain LoRA transfer produces the highest absolute
separability of any arm (full-label reference 59.66% and 10% linear head
45.27%, both above the frozen 58.37%/44.18%), yet its retention falls to
71.98%: the semantic layer cannot harvest the improved geometry.

## Appendix C. Recycling per-seed results and hyperparameter grid

Table C1 lists the per-seed retention of the three headline recycling
configurations. Every NTU120 seed improves; PanAf500 improves on five seeds,
ties on three, and regresses on two (the quantization caveat of §4.8);
UCF101 improves on nine of ten seeds (one seed regresses by 1.94 pp).

| Seed | NTU120 (lr 2e-2, x8) | PanAf500 (lr 1e-2, x4) | UCF101 (lr 2e-2) |
|:-----|:---------------------|:-----------------------|:-----------------|
| 42 | 85.40 | 93.75 | 81.40 |
| 43 | 84.33 | 107.14 | 60.00 |
| 44 | 85.07 | 76.47 | 71.43 |
| 45 | 87.57 | 58.33 | 98.75 |
| 46 | 87.35 | 92.86 | 52.27 |
| 47 | 84.95 | 100.00 | 65.59 |
| 48 | 85.44 | 106.25 | 80.43 |
| 49 | 84.55 | 94.12 | 62.65 |
| 50 | 84.06 | 93.33 | 63.11 |
| 51 | 83.09 | 94.12 | 82.50 |

Per-seed retention is the ratio of the seed's (b)-arm accuracy to its own
full-label reference and can exceed 100% when the (b) arm beats that
reference on a seed; the headline number is the ratio of means.

Bootstrap of the ratio of means (10,000 resamples over seeds) gives 95% confidence intervals of +10.21 pp [+9.44, +11.06] on NTU120, +9.68 pp [+1.27, +16.25] on PanAf500, +20.17 pp [+12.73, +28.24] on UCF101, and +10.82 pp [+9.49, +12.31] on the 120-class replication, and +15.07 pp [+13.63, +16.60] on the 32-channel capacity control, and +8.84 pp [+8.16, +9.49] on the standard-width (64-channel) control. The per-seed sign tests are 10/0/0 ($p=0.002$), 5/3/2 ($p=0.453$), 9/0/1 ($p=0.022$), and 10/0/0 ($p=0.002$) on the 120-class replication, 10/0/0 ($p=0.002$) on the capacity control, and 10/0/0 ($p=0.002$) on the standard-width control; PanAf500's sign test is not significant, so its CONFIRMS verdict is descriptive rather than a significance claim. In raw (b)-arm accuracy the corresponding intervals are +5.85 pp [+5.04, +6.72], +4.00 pp [-2.40, +10.80], +3.63 pp [+2.03, +5.14], and +3.65 pp [+2.99, +4.40] on the 120-class replication, +12.27 pp [+11.40, +13.17] on the capacity control, and +7.68 pp [+7.06, +8.32] on the standard-width control. On the class-addition episode (60->120; Table 8 of the main text) the raw gain is +5.20 pp [+4.70, +5.68], the new-class gain is +7.57 pp [+7.10, +8.05], the retention gain is +15.49 pp [+14.54, +16.55] under the own-denominator convention (+12.11 pp [+10.94, +13.22] under a frozen denominator), and the old-class forgetting is reduced by 2.84 pp [2.03, 3.53]; all sign tests are 10/0/0 (p=0.002).

Table C2 gives the full NTU120 sensitivity grid. The augmentation rows use
50 epochs and the no-augmentation rows 100 epochs, as in the main text;
every cell uses ten seeds. The lr 4e-2 cell terminates at the calibration
gate (Appendix D) and therefore has no retention value.

| Configuration | lr | V | Epochs | lambda | Rank | Retention (%) | Verdict |
|:--------------|:---|:---|:-------|:-------|:-----|:--------------|:--------|
| lr 1e-3 | 1e-3 | 1 | 100 | 1.0 | 8 | 78.87 | FAILS |
| lr 1e-2 | 1e-2 | 1 | 100 | 1.0 | 8 | 82.13 | FAILS |
| lr 2e-2 | 2e-2 | 1 | 100 | 1.0 | 8 | 82.73 | FAILS |
| lr 4e-2 | 4e-2 | 1 | 100 | 1.0 | 8 | gate | tau = 0 |
| aug x4 | 2e-2 | 4 | 50 | 1.0 | 8 | 83.41 | FAILS |
| aug x8 | 2e-2 | 8 | 50 | 1.0 | 8 | 85.16 | PARTIAL |
| aug x12 | 2e-2 | 12 | 50 | 1.0 | 8 | 84.38 | FAILS |
| aug x4, lr 1e-2 | 1e-2 | 4 | 50 | 1.0 | 8 | 82.06 | FAILS |
| epochs 200 | 1e-3 | 1 | 200 | 1.0 | 8 | 78.88 | FAILS |
| no distillation | 1e-3 | 1 | 100 | 0.0 | 8 | 79.05 | FAILS |
| rank 16 | 1e-2 | 1 | 100 | 1.0 | 16 | 81.49 | FAILS |

Table C3 gives the per-seed detail of the 120-class scale replication. Every
seed improves in both raw (b)-arm accuracy and retention; per-seed retention
ranges from 74.39% to 82.61%.

| Seed | Frozen b-arm (%) | Recycling b-arm (%) | Recycling retention (%) |
|:-----|:-----------------|:--------------------|:------------------------|
| 42 | 29.71 | 32.46 | 77.51 |
| 43 | 29.79 | 33.75 | 77.51 |
| 44 | 28.75 | 35.04 | 82.61 |
| 45 | 28.96 | 33.17 | 74.74 |
| 46 | 30.29 | 33.21 | 76.20 |
| 47 | 28.63 | 31.58 | 74.39 |
| 48 | 28.96 | 32.92 | 77.91 |
| 49 | 29.04 | 33.50 | 78.21 |
| 50 | 29.46 | 31.46 | 74.60 |
| 51 | 30.50 | 33.46 | 78.11 |

Table C4 repeats the per-seed detail for the 32-channel capacity control. Every
seed improves in retention; the frozen b-arm mean is
38.93% and the recycled
mean is 51.20%.

| Seed | Frozen b-arm (%) | Recycling b-arm (%) | Recycling retention (%) |
|:-----|:-----------------|:--------------------|:------------------------|
| 42 | 38.43 | 51.19 | 83.23 |
| 43 | 38.45 | 49.86 | 81.07 |
| 44 | 40.51 | 51.75 | 83.49 |
| 45 | 39.27 | 52.80 | 85.99 |
| 46 | 38.75 | 53.40 | 85.57 |
| 47 | 37.16 | 50.65 | 84.78 |
| 48 | 39.67 | 53.23 | 87.89 |
| 49 | 39.67 | 50.40 | 82.43 |
| 50 | 38.88 | 49.05 | 79.80 |
| 51 | 38.54 | 49.65 | 81.29 |

Table C5 repeats the per-seed detail for the standard-width (64-channel)
control. Every seed improves in retention; the frozen b-arm mean is
44.63% and the recycled
mean is 52.31%.

| Seed | Frozen b-arm (%) | Recycling b-arm (%) | Recycling retention (%) |
|:-----|:-----------------|:--------------------|:------------------------|
| 42 | 43.92 | 51.02 | 82.58 |
| 43 | 45.04 | 52.58 | 84.35 |
| 44 | 45.10 | 53.40 | 85.27 |
| 45 | 45.04 | 53.42 | 84.67 |
| 46 | 43.64 | 53.44 | 84.05 |
| 47 | 44.35 | 51.96 | 84.13 |
| 48 | 44.61 | 52.73 | 84.21 |
| 49 | 44.35 | 51.36 | 82.70 |
| 50 | 44.74 | 51.88 | 81.97 |
| 51 | 45.51 | 51.30 | 82.26 |

Table C6 gives the per-seed detail of the class-addition episode (60->120;
old anchors stay spent). Forgetting is the phase-1 old-class accuracy minus
the phase-2 old-class accuracy.

| Seed | Phase1 old (%) | Semantic all (%) | Recycling all (%) | New sem. (%) | New rec. (%) | Forget. sem. (pp) | Forget. rec. (pp) |
|:-----|:---------------|:-----------------|:------------------|:-------------|:-------------|:------------------|:------------------|
| 42 | 44.17 | 28.21 | 32.25 | 22.17 | 30.00 | 9.92 | 9.67 |
| 43 | 41.50 | 25.54 | 29.50 | 21.25 | 27.75 | 11.67 | 10.25 |
| 44 | 42.33 | 26.50 | 32.46 | 20.75 | 29.00 | 10.08 | 6.42 |
| 45 | 43.08 | 27.62 | 32.21 | 22.58 | 30.17 | 10.42 | 8.83 |
| 46 | 41.58 | 27.17 | 33.37 | 22.83 | 31.75 | 10.08 | 6.58 |
| 47 | 42.50 | 28.33 | 33.87 | 25.00 | 32.58 | 10.83 | 7.33 |
| 48 | 43.92 | 26.96 | 32.08 | 23.00 | 30.00 | 13.00 | 9.75 |
| 49 | 41.25 | 28.00 | 33.37 | 23.50 | 30.58 | 8.75 | 5.08 |
| 50 | 43.83 | 26.67 | 32.92 | 22.50 | 30.92 | 13.00 | 8.92 |
| 51 | 44.75 | 28.08 | 33.08 | 22.67 | 29.17 | 11.25 | 7.75 |

## Appendix D. Calibration-gate failure cases

The calibration gate turns a silent quality collapse into an explicit
error. Three configurations exceed the adaptation envelope and all stop in
the same place. On NTU120 at lr 4e-2, the anchor objective never leaves
chance: the loss stays at 5.10 for 100 epochs and the anchor accuracy ends
at 1.7% over 60 classes. On PanAf500 at lr 2e-2 with 4x augmentation, the
anchor objective does fit (loss 5.51 to 1.75, anchor accuracy 35.7% over 9
classes), but the adapted feature geometry still drives the selected
similarity threshold to zero. In the 32-channel capacity control, the
16-channel-optimal recipe (lr 2e-2, 8x augmentation) completes five seeds and
trips on the sixth (seed 47): the anchor objective stays at chance (loss
5.10, anchor accuracy 1.7% over 60 classes). In all cases the semantic stage
raises `ValueError: tau must be positive: 0.0` from the threshold constructor
before any pseudo-label is emitted.

The three cases are complementary. The first shows a gate trip caused by an
objective that never converged; the second shows a gate trip caused by a
converged objective whose geometry is nevertheless incompatible with the
semantic layer's calibration. The third shows that the upper edge moves with
encoder width: the recipe that is stable at 16 channels is too strong at 32,
so the capacity control uses lr 1e-2, where all ten seeds pass. A
training-loss monitor alone would have
accepted the second run. The gate is therefore not a duplicate of the
training objective but an independent check on the feature geometry that
the next stage will actually consume.

## Appendix E. Reproduction

The three-arm measurement is implemented once, in
`labelfree_tier_replication.py` (`run_arms`), and reused by every table:
arm (a) fits a linear head on the `b`% anchors; arm (b) runs the semantic
pipeline on the same anchors; arm (c) fits the full-label linear reference
on the same frozen features. The recycling driver
`t4_budget_split_tier.py` performs, per seed, the LoRA update on that seed's
anchors, re-exports the features, and re-runs the same three arms, so the
denominator always reflects the encoder actually measured.

```bash
# 1) feeder export (T=30, 25-slot map)
python scripts/p1_sprint/export_tier_feeder.py --tier ntu120

# 2) frozen label-free pretext (150 epochs) and three-arm readout
python scripts/p1_sprint/labelfree_tier_replication.py --tier ntu120 --seeds 10

# 3) anchor recycling (one command per configuration; tag fixes the summary name)
python scripts/p1_sprint/t4_budget_split_tier.py --tier ntu120 --arm same10 \
  --seeds 10 --epochs 100 --lr 0.02 --aug-views 8 --tag aug8_lr2e2

# 4) label-free baselines
python scripts/p1_sprint/tta_tent_tier.py --tier ntu120 --seeds 10 --bn-mode eval
python scripts/p1_sprint/lora_continue_tier.py --tier ntu120 --epochs 30 --lr 0.01 \
  --source-ckpt runs/p1_sprint/labelfree_ntu120/pretext/epoch150_model.pt
python scripts/p1_sprint/supcon_feedback_tier.py --tier ntu120 --rounds 1 --epochs 20 \
  --psd-seed 42

# 5) 120-class scale replication (balanced 120 x 100/20, fresh pretext)
python scripts/p1_sprint/export_tier_feeder.py --tier ntu120_full
python scripts/p1_sprint/labelfree_tier_replication.py --tier ntu120_full --seeds 10
python scripts/p1_sprint/t4_budget_split_tier.py --tier ntu120_full --arm same10 \
  --seeds 10 --epochs 50 --lr 0.02 --aug-views 8 --tag full10

# 6) 32-channel capacity control (same feeder via directory junction)
python scripts/run_ntu_phaseb.py --config configs/labelfree_ntu120_32ch_pretext.yaml
python scripts/p1_sprint/labelfree_tier_replication.py --tier ntu120_32ch --seeds 10
python scripts/p1_sprint/t4_budget_split_tier.py --tier ntu120_32ch --arm same10 \
  --seeds 10 --epochs 50 --lr 0.01 --aug-views 8 --tag aug8_lr1e2

# 7) standard-width (64-channel) control
python scripts/run_ntu_phaseb.py --config configs/labelfree_ntu120_64ch_pretext.yaml
python scripts/p1_sprint/labelfree_tier_replication.py --tier ntu120_64ch --seeds 10
python scripts/p1_sprint/t4_budget_split_tier.py --tier ntu120_64ch --arm same10 \
  --seeds 10 --epochs 50 --lr 0.01 --aug-views 8 --tag aug8_lr1e2

# 8) class-addition taxonomy episode (60 -> 120)
python scripts/p1_sprint/taxonomy_addition_recycle.py --seeds 10 \
  --epochs 50 --lr 0.02 --aug-views 8 --tag add60to120
```

Checkpoints: the frozen pretexts are
`runs/p1_sprint/labelfree_<tier>/pretext/epoch150_model.pt`; the NTU60 source
used by the cross-domain arms is
`runs/ntu_phaseB/joint_pretext/epoch300_model.pt`; the released adapted
checkpoints are `adapt_lora_xfer_ntu120/lora_epoch30_model.pt` and
`adapt_supcon_ntu120/supcon_r1_epoch20_model.pt`. Every summary JSON carries
its seeds, per-seed rows, configuration, and wall-clock time; the evidence
mirror under `reports/p1_sprint_evidence/` is byte-identical to the run
outputs.

<!-- APPENDIX:END -->


















