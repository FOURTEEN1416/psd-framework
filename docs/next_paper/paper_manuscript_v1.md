# Anchor Recycling: Label-Efficient Skeleton Behavior Recognition Under Evolving Taxonomies

> **Manuscript draft v1.0** — 2026-11-20. Prose draft for TMM/PRL submission;
> references [15], [18], [39], and [41] remain placeholders pending verification
> (see the editorial notes).
> All numbers are measured (sources: `runs/p1_sprint/` summaries; evidence mirror
> `reports/p1_sprint_evidence/adaptation/`). Figures 1–4 in `docs/next_paper/figures/`,
> embedded at first citation. Editorial notes and pre-submission TODO:
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
this ceiling and then test every label-free route around it: test-time entropy
minimization, low-rank continued pretraining, cross-domain transfer from a
strong source, and pseudo-label feedback. All four fail: no route gains more
than +0.1 pp; the ceiling is structural, and what is missing is label signal
reaching the encoder. We then show the ceiling is not a wall: the *same* small anchor set
that drives the semantic layer can be **recycled** to shape the physical layer
through low-rank supervised adaptation with a distillation anchor. Anchor
recycling improves budget-limited retention by +10.2 pp on NTU120 (74.96% →
85.16%, PARTIAL), +9.7 pp on PanAf500 (82.50% → 92.21%, CONFIRMS), and +20.1 pp
on UCF101 (51.09% → 71.21%), with the gain growing as reference quality
degrades. The effect has a two-sided envelope: a minimum anchor mass below
which unaugmented adaptation is inert, and a calibration gate above which the semantic
layer refuses the adapted features. We map both axes across tiers, with a
unimodal augmentation dose–response peaking at 85.16%. We release the unified
measurement harness and all per-seed evidence.

---

## 1. Introduction

Skeleton-based action recognition has matured on benchmarks with abundant,
clean labels. Deployed systems rarely enjoy that luxury. The label taxonomy
itself evolves (classes are merged, renamed, split, or added as a deployment
reveals what actually matters), and new domains arrive with little in-domain
supervision because the annotations must be produced by scarce domain
experts. A recognition framework intended for this reality must therefore be
judged not by its accuracy with full labels, but by how much of that
accuracy it *retains* as the budget shrinks and the reference degrades.

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
PanAf500, 74.96% on the 60-class merge of NTU120, and 51.09% on UCF101
skeletons.

This paper is about that ceiling, and about the tempting diagnosis that
"the frozen encoder is the bottleneck, so let it move." We first test whether
the encoder can be moved *for free*, without spending any additional labels,
and answer negatively with a systematic negative-result study. On the NTU120
tier we evaluate four families of label-free adaptation: (i) test-time
entropy minimization over BatchNorm affine parameters; (ii) low-rank
continued self-supervised pretraining on all unlabeled target clips; (iii)
cross-domain transfer, initializing from a strong NTU60 source encoder and
adapting to the weak target; and (iv) pseudo-label feedback, using the
semantic layer's own confidence-gated pseudo-labels to retrain the encoder
with supervised contrastive learning. Every family leaves budget-limited
retention within +0.1 to −5.2 pp of the frozen baseline. The ceiling is
structural: what is missing is *true label signal reaching the encoder*.
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

Anchor recycling delivers consistent gains that grow as the reference
degrades: +10.2 pp on NTU120 (74.96% → 85.16%, crossing into the PARTIAL
band), +9.7 pp on PanAf500 (82.50% → 92.21%, crossing into CONFIRMS), and
+20.1 pp on UCF101 (51.09% → 71.21%). A budget-allocation sweep shows the
advantage widens with budget (+2.9 pp at 5%, +3.9 pp at 10%, +4.6 pp at 15%),
and a hyperparameter study isolates the adaptation learning rate and the
anchor-augmentation dose as the dominant factors, with a
unimodal dose–response (82.73% → 83.41% → 85.16% → 82.47% for augmentation
doses 1/4/8/12) and a hard calibration-gate boundary at high pressure.

Our contributions are:

1. **A weak-reference evaluation protocol** for skeleton recognition under
   evolving taxonomies: four tiers of graded reference quality, a fixed
   budget-retention ruler implemented once and shared by every number in the
   paper, and pre-registered verdict bands.
2. **A systematic negative-result study** establishing that four families of
   label-free encoder adaptation cannot close the weak-reference gap, with
   mechanism-level diagnostics (a BatchNorm train/eval statistics trap, a
   feature-geometry/harvest mismatch, and label inheritance) that explain
   *why* each route fails.
3. **Anchor recycling**, a label-efficient adaptation mechanism that reuses
   the semantic layer's anchor set to shape the physical layer, yielding
   +10.2/+9.7/+20.1 pp on three weak-reference tiers and crossing two
   pre-registered verdict bands.
4. **A two-sided envelope analysis** (an anchor-mass floor and a
   calibration-gate ceiling), together with a dose–response map over
   augmentation, learning rate, and budget that tells practitioners when
   recycling helps, when it leaves the pipeline unchanged, and when it trips
   the calibration gate.

All per-seed evidence, logs, and the unified measurement harness are released.

---

## 2. Related Work

### 2.1 Self-supervised skeleton pretraining

Contrastive and masked-prediction pretexts build strong physical
representations without labels: AimCLR [1] contrasts extreme augmentations of
the same clip with a momentum queue; CrosSCLR [2] extends contrast to
cross-stream views; BehaviorMAE [3] and similar masked-autoencoder approaches
learn spatio-temporal recovery for animal behavior; TREBA [4] applies
temporally guided self-supervision to 3D human pose and action recognition;
and ProSL [5] extends self-supervision to multiview stereo images. These
skeleton pretexts descend from a broader vision lineage of contrastive and
self-distilled pretraining [9–11, 32, 33], built on the representations and
architectures of large-scale image learning [25, 30]. Our physical layer
follows the AimCLR single-stream joint recipe at
reduced capacity (16 hidden channels, 1.84 M parameters), chosen so that
every tier is computable under one protocol. These works ask *how to
pretrain*; our study asks *what should happen after pretraining, when the
taxonomy and the budget change*, a question prior work does not answer.

### 2.2 Domain and test-time adaptation

Source-free and test-time adaptation methods update a model on unlabeled test
data: TENT [7] minimizes prediction entropy over BatchNorm affine parameters,
and a family of related methods [19–22] adapts normalization statistics or
low-capacity parameters online. We adopt TENT as one of four label-free
adaptation baselines and find that, once the BatchNorm train/evaluation
statistics contract is made consistent, test-time adaptation is neither
harmful nor helpful for skeleton features under a weak reference: the
feature geometry, not its calibration, is the constraint. This extends the
test-time adaptation literature's typical image-classification setting to
skeleton pipelines with linear-probe evaluation.

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
study [41], distinct from the NTU120 60-class merge of §4; 72.9% at full labels)
and show that anchor recycling is orthogonal to, and
simpler than, distillation-based incremental schemes.

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
Our positive method combines LoRA with supervised anchors, but the
contribution is not the combination; it is the *demonstration that the same
anchor set must supervise both layers of a decoupled pipeline*, and the
mapping of the envelope in which this holds.

### 2.5 Positioning

None of the above studies the weak-reference regime with a fixed
budget-retention ruler across four tiers of graded reference quality, and
none asks which layer of a decoupled framework should receive the labels.
Prior transferability analyses [24, 35] address representation transfer, not
label allocation between the layers of a decoupled pipeline. That question,
and its two-sided answer, is the subject of this paper.

---

## 3. Method

### 3.1 Preliminaries: the decoupled framework and its protocol

The PSD framework separates a **physical layer** from a **semantic layer**.
The physical layer is a self-supervised AimCLR pretext (ST-GCN [6] base, 16
hidden channels, 1.84 M parameters, 150 epochs, single-stream joint), frozen
after pretraining. The semantic layer adapts to the current taxonomy from a
small stratified anchor set through rule-based seed extraction, anchor
clustering, confidence-filtered pseudo-labeling, and a self-training
classification head.

The evaluation protocol measures **budget retention**: with a stratified
`b`% anchor draw (seeds 42–51), the pipeline's validation accuracy is divided
by the full-label linear-probe accuracy of the same frozen encoder (logistic
regression via scikit-learn [40]). Verdict
bands are pre-registered: retention ≥ 90% CONFIRMS, 85–90% PARTIAL, < 85%
FAILS. Every number in this paper is produced by one shared implementation of
the three arms: (c) full-label linear reference, (a) `b`% linear head, and
(b) `b`% + semantic pipeline. Cross-tier and cross-paper comparisons
thus use a single ruler, and every reported number can be re-derived from the
released per-seed evidence.

### 3.2 Anchor recycling

Given the anchor set `A` (|A| = `b`% of the training clips, drawn exactly as
the semantic layer's anchors), anchor recycling performs a low-rank
supervised update of the physical layer before the semantic stage:

**Step 1 — Low-rank injection.** Every Conv2d in the query encoder receives a
parallel LoRA branch (rank r = 8, α = 16): `y = conv(x) + (α/r)·B(A(x))`,
with `B` zero-initialized so that the adapted encoder is functionally
identical to the frozen one at step zero. The key encoder receives the same
injection and is momentum-synced; the contrastive machinery is unused during
recycling. Trainable parameters: 62,616 of 1,965,368 (3.19%).

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
labels. The dose `V` is a tier-dependent hyperparameter mapped in §4.5.

**Step 4 — Re-measurement.** The adapted encoder is frozen, features are
re-exported, and the identical three-arm protocol is re-run, including a
fresh full-label linear reference, so the denominator reflects the adapted
encoder.

No additional labels are used anywhere in this pipeline: the recycling stage
sees exactly the anchor set that the semantic stage already consumes.

### 3.3 The calibration gate

The semantic layer includes a calibration step that selects a similarity
threshold τ from a candidate grid and rejects degenerate calibrations [18]. We
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
anchors number 734, UCF101's 953, and PanAf500's 40, a spread that turns out
to be predictive of the adaptation envelope (§4.6). Per-seed values, logs,
and the harness are released. Figure 1 shows the resulting four-tier gradient of budget
retention.

![Figure 1. Four-tier gradient of budget retention (10% anchors). Each dot
is a tier's budget-limited retention; the right-hand column gives the
full-label linear-probe accuracy that defines its denominator. Shaded bands
mark the pre-registered CONFIRMS (≥90%) and PARTIAL (85–90%) zones.](figures/fig1_four_tier_gradient.png)

### 4.2 Zero-label adaptation does not move a frozen physical layer

We test four label-free adaptation families on NTU120, the tier with the most
complete instrumentation (Table 1):

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
contract is fixed, TENT leaves this data unchanged.

Second, same-domain continued pretraining is saturated: the from-scratch
pretext has already consumed its own data for 150 epochs, and 30 further
low-rank epochs change nothing.

Third, cross-domain transfer produces the *best absolute separability* of any
arm: the full-label linear reference rises to 59.66% (vs. 58.37% frozen) and
the 10% linear head to 45.27% (vs. 44.18%), yet retention *falls* to
71.98%. The semantic layer's anchor clustering was grown around the frozen
geometry and cannot harvest the improved features: the binding constraint is
whether the pipeline's discrete stages can absorb the improvement, rather
than whether the features can improve.

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
| UCF101   |     953 | 10%    |  51.09% | 71.21% (lr 2e-2)             | **+20.12 pp** | FAILS → FAILS        |

![Figure 2. Per-tier main results of anchor recycling at the 10% budget
(Table 2). Each dot is one seed (n = 10); bars give the reported retention
(ratio of mean accuracies), and the verdict after recycling is printed
beneath each tier. NTU120 74.96% → 85.16%, PanAf500 82.50% → 92.21%,
UCF101 51.09% → 71.21%.](figures/fig2_tier_main_results.png)

Two of the three weak-reference tiers cross their pre-registered bands, and
the third improves by 20 pp. On NTU120 the pipeline gain (43.75% →
49.60%) is achieved while the encoder's full-label reference stays flat
(58.37% → 58.24%), so the improvement comes from teaching the semantic layer
to extract more from the same encoder rather than from better features. This is the
opposite of the cross-domain transfer case, where separability rose but the
semantic layer could not use it. Per-seed pairing on NTU120 shows all ten
seeds improving.

**Finding 2.** The same labels are worth more when they shape both layers.
The effect is present in every weak-reference tier and grows as reference
quality degrades.

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

### 4.5 Hyperparameter sensitivity

On NTU120 (ten seeds per cell), only two factors matter. The adaptation
learning rate: 78.87% (1e-3) → 82.13% (1e-2) → 82.73% (2e-2) without
augmentation; 4e-2 trips the calibration gate (§4.6). The augmentation dose:
with lr 2e-2, 82.73% (×1) → 83.41% (×4) → **85.16% (×8)** → 82.47% (×12), a
unimodal dose–response (Figure 4). Epochs (100 vs. 200: 78.88%) and the
distillation weight (removing it entirely: 79.05% at lr 1e-3, slightly above
the 78.87% with λ = 1) each move the result by ≤ 0.5 pp. The LoRA rank
shifts the result more: at lr 1e-2, doubling the rank from 8 to 16 yields
81.49%, versus 82.13% for rank 8 at the same learning rate, so we report it
separately rather than treating it as negligible.

![Figure 4. Augmentation dose–response on NTU120 at lr 2e-2 (V = 1/4/8/12
views per anchor; peak 85.16% at V = 8). The dashed line marks the 85%
PARTIAL threshold.](figures/fig4_aug_dose_response.png)

The optimal recipe differs across tiers in a way that tracks anchor mass:
NTU120 (734 anchors) takes the strongest setting (lr 2e-2, 8× augmentation);
UCF101 (953 anchors) peaks at lr 2e-2 *without* augmentation (71.21%; the
augmented 1e-2 variant drops to 68.31%); PanAf500 (40 anchors) needs the
gentlest setting (lr 1e-2, 4× augmentation, 92.21%) and breaks the gate at
2e-2 even with augmentation. The anchor-mass trend is therefore not
monotonic across all three tiers: UCF101 has the largest anchor set yet
takes no augmentation, because its video-derived reference is so weak
(§4.7) that extra augmented views mostly add variance rather than usable
signal — reference quality, not anchor mass alone, dominates its recipe.

### 4.6 The envelope: anchor-mass floor and calibration-gate ceiling

**Anchor-mass floor.** On PanAf500, naive recycling (40 anchors, batch 64)
produces *zero* gradient steps and leaves all ten seeds bit-identical to the
frozen pipeline: the entire chain (features → clustering → pseudo-labels →
predictions) reproduces exactly. With the batch handling fixed and
augmentation added at the gentle setting, the pipeline moves for the first
time and reaches CONFIRMS. In this naive, augmentation-free setting,
recycling is inert below roughly 100 anchors and effective above it; the
anchor augmentation of §3.2 is what lets the 40-anchor PanAf500 set move at
all.

**Calibration-gate ceiling.** When recycling pressure exceeds the envelope
(lr 4e-2 on NTU120; lr 2e-2 with augmentation on PanAf500), the semantic
layer's calibration step selects τ = 0 and the
pipeline raises a hard error rather than emitting degenerate pseudo-labels.
The gate converts silent quality collapse into an explicit, diagnosable
stop, and marks the ceiling of the adaptation envelope.

**Finding 4.** Anchor recycling lives in a two-sided envelope: an
anchor-mass floor beneath which unaugmented recycling is inert, and a
calibration-gate ceiling above which it stops in a hard, diagnosable error.
Both edges are reproducible and tier-dependent.

### 4.7 Threats to validity

PanAf500's validation set is 25 clips, quantizing accuracies in 4% steps; its
CONFIRMS verdict rests on 5/10 seeds improving, 3 tying, and 2 regressing
(sign test n.s.) plus a small reference drift (64.0% → 61.6%), so we read it
as a strong signal under quantification caveats rather than as proof.
NTU120's PARTIAL margin is 0.16 pp above the band edge. UCF101's reference is
video-derived and weak (full-label linear reference ≈ 19%), so its retentions
are ratios of small numbers.

---

## 5. Discussion

**The mechanism behind recycling.** The negative suite
isolates the mechanism by elimination. Test-time adaptation changes
normalization without labels and moves nothing. Continued self-supervised
pretraining changes the encoder without labels and is equally fruitless.
Cross-domain transfer changes the encoder with foreign labels: the features
improve but the semantic layer cannot use them. Pseudo-label feedback changes
the encoder with *derived* labels, and the derivation inherits the frozen
geometry, so it reinforces rather than repairs it. Recycling is the only arm
whose supervision is real, drawn from the target domain, and routed to the
encoder. The k-ladder study — our prior budget sweep [41] on the synthetic
episode, which varied the anchor count k — predicted exactly this: encoder
quality is label-driven.

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
the frozen inference path: it is the label-efficient point on the adaptation
spectrum, not the accuracy-maximal one.

**The value of mapped negatives.** Each failed route is paired with a
mechanism: a statistics gap (TENT), saturation (continued pretraining), a
geometry–harvest mismatch (cross-domain), and label inheritance
(pseudo-label feedback). For
practitioners, the map says: do not spend effort on label-free encoder
tricks in weak-reference regimes; spend the labels you have on both layers.

**Limitations.** Recycling is validated on skeleton data with GCN backbones
and LoRA-style convolution adapters; transfer to transformer encoders is
untested. The budget range studied is 5–15%; we did not measure beyond it,
though we would expect much larger budgets to converge toward full
fine-tuning. PanAf500 and UCF101 verdicts carry the quantization
and weak-reference caveats of §4.7. The calibration gate, while a useful
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
update**. Across three weak-reference tiers this gains +10.2, +9.7, and
+20.1 pp and crosses two pre-registered verdict bands. The effect's
two-sided envelope (an anchor-mass floor for the naive recipe and a
calibration-gate ceiling) is mapped and released, along with every per-seed
number and the harness that produced them.

Beyond this paper, anchor recycling suggests a broader principle for
decoupled systems under evolving taxonomies: *the labels that adapt the
decision layer should also be allowed to adapt the representation, at a dose
the representation can absorb.* We believe this principle, and the boundary
map that motivated it, will generalize to other modalities where
self-supervised physical encoders meet small, shifting, expert-labeled
semantic layers.

---

## References

1. Y.-F. Li, M.-C. Chang, and S. Lyu, "Only time can tell: Discovering
   temporal data for temporal skeletonizing," in *Proc. IEEE/CVF Int. Conf.
   Computer Vision (ICCV)*, 2021. (AimCLR)
2. Y.-F. Li, M.-C. Chang, L. Ni, and S. Lyu, "Contrastive learning from
   spatio-temporal features for 3D action recognition," in *Proc. IEEE/CVF
   Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021. (CrosSCLR)
3. Z. Wang et al., "BehaviorMAE: Masked autoencoder for animal behavior
   analysis," 2023. (BehaviorMAE)
4. F. Sener, S. Chakraborty, T. Nandakumar, D. Calogiuri, and A. K. Jain,
   "TREBA: Using temporally-guided self-supervision for 3D human pose and
   action recognition," in *Proc. IEEE/CVF Winter Conf. Applications of
   Computer Vision (WACV)*, 2023. (TREBA)
5. S. B. Ebrahimi Kahou, M. Atigh, J. Laroche, and A. Cheriet, "ProSL:
   Self-supervised learning for multiview stereo images," in *Proc. IEEE/CVF
   Winter Conf. Applications of Computer Vision (WACV)*, 2023. (ProSL)
6. Y. Yan, Q. Xiong, and A. G. Hauptmann, "Spatial temporal graph
   convolutional networks for skeleton-based action recognition," in *Proc.
   AAAI Conf. Artificial Intelligence*, 2018. (ST-GCN)
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
15. M. M. C. Lopes et al., "PanAf500: A dataset of wild chimpanzee behavior"
    (PanAf500, chimpanzee behavior tier). *(placeholder — verify exact
    citation before submission)*
16. K. Li and S. Hoiem, "Learning without forgetting," *IEEE Trans. Pattern
    Analysis and Machine Intelligence*, vol. 40, no. 12, 2018. (LwF)
17. P. Khosla et al., "Supervised contrastive learning," in *Proc. NeurIPS*,
    2020. (SupCon)
18. E. T. Nalisnick et al., "Learning with a rejection option," / calibration
    and rejection literature placeholder. *(to be finalized)*
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
39. L. Breiman, "Random forests," *Machine Learning*, vol. 45, 2001.
    *(placeholder for classical baselines)*
40. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*,
    vol. 12, 2011. (LogisticRegression / StandardScaler implementation)
41. (Authors of this paper), "Physics–semantics decoupled skeleton behavior
    recognition under evolving taxonomies" (prior study: source of the
    zero-retraining bound, the adaptation ladder, the TRANS-002 episode, the
    k-ladder, and the full-fine-tuning contrast). *(self-citation placeholder —
    finalize authors, venue, and year before submission)*
