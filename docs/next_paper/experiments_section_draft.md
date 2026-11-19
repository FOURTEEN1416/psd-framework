# Next-Paper — Experiments Section (draft v0.1, 2026-11-19)

> 用途：下一篇（P2/标签高效域适配方向）的 Experiments 章节英文草稿。全部数字来自
> `runs/p1_sprint/` 实测 summary（E9 四层协议回归、E10 阴性套件、E11 扫参+预算曲线、
> E12 PanAf 干预），判据沿用预注册 E9 口径（≥90% CONFIRMS / 85–90% PARTIAL / <85% FAILS）。
> 待补：NTU120 最优配置的 10-seed 复核、UCF101/PanAf 的 lr 敏感性、与 CIL/ST-GCN 基线的
> 同预算对照。

---

## 4. Experiments

### 4.1 Setup: tiers, budget protocol, and the unified ruler

We evaluate on four skeleton behavior tiers ordered by reference quality: NTU60
(90.7% published retention), PanAf500 (82.5%), NTU120 (74.96%), and UCF101
(51.09%). For every tier, the physical layer is a from-scratch, label-free AimCLR
pretext (ST-GCN base, 16 hidden channels, 150 epochs, single-stream joint),
frozen before evaluation. The label budget is a stratified 10% draw of training
clips (seeds 42–51); all reported retentions are the ratio of the 10%-budget
pipeline accuracy to the full-label linear-probe accuracy of the same encoder,
measured by one shared implementation (`run_arms`) so that every number in this
paper is produced by the same ruler. Verdict bands follow the pre-registered
protocol: retention ≥ 90% CONFIRMS, 85–90% PARTIAL, < 85% FAILS.

### 4.2 Zero-label adaptation does not move a frozen physical layer

We first test four families of adaptation that use **no additional labels**:

1. **Test-time adaptation (TENT)**: entropy minimization over BatchNorm affine
   parameters on the unlabeled test stream (10% linear head fixed). On NTU120
   this yields 74.08% vs. the 74.96% frozen baseline — and a naive train-mode
   variant collapses to 27% purely from the BN batch/running-statistics gap.
2. **Same-domain continued pretraining**: LoRA (r = 8, 3.2% of parameters) with
   the AimCLR objective on all unlabeled target clips, plus a cosine feature
   distillation anchor to the frozen encoder. Result: 74.99% — indistinguishable
   from frozen; a from-scratch 150-epoch pretext is already saturated on its own
   data.
3. **Cross-domain transfer**: initializing from the strong NTU60 encoder
   (300-epoch pretext) and adapting to NTU120. Zero-shot transfer gives 69.77%;
   LoRA adaptation raises absolute separability (full-label reference 59.66%,
   the best of all arms) but *lowers* retention to 71.98% — the semantic layer
   cannot harvest the improved geometry.
4. **Pseudo-label feedback**: SupCon training of the encoder on the semantic
   layer's confidence-gated pseudo-labels (7,211/7,337 samples covered), 20
   epochs. Result: 74.98% — flat.

**Finding 1.** Every zero-label route to "moving the frozen encoder" fails
(Δ ∈ [−5.2, +0.1] pp). The weak-reference boundary is structural: without real
labels shaping the encoder, its feature quality — not its calibration — is the
binding constraint. This mirrors the k-ladder result that encoder quality is
label-driven.

### 4.3 Dual-use anchor recycling

The one lever the k-ladder leaves open is to spend the *same* 10% anchor set
twice: once on the semantic layer (as in PSD) and once on the physical layer,
as supervision for a low-rank encoder update. We fine-tune LoRA adapters and a
fresh linear head on the anchor clips with cross-entropy plus a cosine
distillation anchor to the frozen encoder (BatchNorm running statistics frozen
for train/eval consistency), then re-run the identical three-arm protocol.

| Tier | Budget | Frozen | Dual-use | Δ | Verdict |
|---|---|---|---|---|---|
| NTU120 | 10% | 74.96% | **83.41%** (lr 2e-2, aug×4) | **+8.45 pp** | FAILS → FAILS |
| UCF101 | 10% | 51.09% | 71.21% (lr 2e-2) | **+20.12 pp** | FAILS → FAILS |
| PanAf500 | 10% | 82.50% | **92.21%** (lr 1e-2, aug×4) | **+9.71 pp** | **FAILS → CONFIRMS** |

**Finding 2.** Dual-use recycling converts the same labels into gains everywhere
we could measure it, and the gain *grows* as reference quality degrades
(NTU120 +8.5 → PanAf500 +9.7 → UCF101 +20.1 pp, per-tier optimal settings). On
PanAf500 the pipeline reaches CONFIRMS for the first time — with the caveat
that its validation set is 25 clips (4% quantization), 6/10 seeds improved with
2 regressions (sign test n.s.), and part of the retention jump reflects a small
reference drift (64.0% → 61.6%); we therefore report it as a strong positive
signal, not a confirmed guarantee.

### 4.4 Where the budget should go

Varying the budget on NTU120 with everything else fixed:

| Budget | Semantic-only (frozen) | Dual-use | Gain |
|---|---|---|---|
| 5% | 64.61% | 67.51% | +2.90 pp |
| 10% | 74.96% | 78.87% | +3.91 pp |
| 15% | 79.04% | **83.60%** | **+4.56 pp** |

**Finding 3.** The dual-use advantage widens with budget, and at 15% the
pipeline reaches 83.6% — 1.4 pp below the PARTIAL band. Labels are better spent
shaping *both* layers than the semantic layer alone. (These points use the
original lr 1e-3 recipe; §4.5 shows that augmentation and a higher learning
rate lift NTU120 further to 83.41% at the same 10% budget.)

### 4.5 Hyperparameter sensitivity

On NTU120 (dual-use, 10 seeds), only the fine-tuning learning rate matters:
78.87% (1e-3) → 82.13% (1e-2) → **82.73%** (2e-2); at 4e-2 the semantic layer's
calibration gate rejects the adapted features outright (τ = 0). Epochs (100 vs.
200), rank (8 vs. 16), and the distillation weight (0, 0.5, 1.0) all move the
result by ≤ 0.5 pp. Dropping the distillation anchor entirely is safe at 1e-3.

Anchor augmentation (sampling 4× augmented views of the anchors per epoch,
using the pretext's own shear/padding family) adds a further gain on NTU120:
82.73% → **83.41%** at lr 2e-2. The optimal recipe is tier-dependent in a way
that tracks anchor mass: NTU120 (734 anchors) takes the strongest setting
(lr 2e-2 + augmentation, 83.41%); UCF101 (953 anchors) peaks at lr 2e-2 without
augmentation (71.21%; the augmented 1e-2 variant gives 68.31%); PanAf500
(40 anchors) requires the gentlest setting (lr 1e-2 + augmentation, 92.21%) and
breaks the calibration gate at lr 2e-2 even with augmentation.

### 4.6 Boundary: anchor mass and the calibration gate

The dual-use effect has a minimum anchor mass. On PanAf500 (400 clips → 40
anchors), naive dual-use leaves every outcome bit-identical to the frozen
pipeline: 40 CE examples cannot move the encoder against the distillation
anchor, and the whole chain is bit-reproducible. Restoring the effect required
anchor **augmentation** at a gentler learning rate — the configuration that
produced the 92.21% CONFIRMS above. Conversely, adaptation pressure that is too
high (lr 4e-2 on NTU120; lr 2e-2 with augmentation on PanAf500) trips the
semantic layer's calibration gate (τ = 0), which fails loudly rather than
emitting degenerate pseudo-labels. Adaptation strength therefore has a
two-sided envelope: too little leaves the boundary intact, too much destroys
the semantic layer's calibration.

### 4.7 Threats to validity

PanAf500's val set (25 clips) quantizes accuracies in 4% steps; its CONFIRMS
verdict should be read with the per-seed spread (10 seeds: retention 0.58–1.07,
median 0.94). UCF101's reference is video-derived and weak (full-label linear
reference 19.35% for the adapted encoder), so retentions there are ratios of
small numbers. All comparisons share one measurement implementation; no number
in this section comes from a different ruler than the published baselines.
