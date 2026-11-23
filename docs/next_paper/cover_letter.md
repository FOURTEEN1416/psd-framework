# Cover Letter — Anchor Recycling

> 使用说明：`[JOURNAL]`/`[EDITOR]` 按目标期刊替换；三个定制段（TMM / PRL / CVIU）
> 在文末，投哪本就把对应段落放进正文第 3 段并删除其余两段。

---

Dear [EDITOR],

We are pleased to submit our manuscript, **"Anchor Recycling: Label-Efficient
Skeleton Behavior Recognition Under Evolving Taxonomies,"** for consideration
as a [article type] in *[JOURNAL]*.

**The problem.** In deployed skeleton-based behavior recognition, label
taxonomies keep evolving (classes are merged, split, or added) and new domains
arrive with only a small annotation budget, while most methods assume a fixed
class set and abundant labels.

**Our contribution.** We study a decoupled framework with a frozen
self-supervised physical layer and a semantic layer that adapts to the current
taxonomy from a small anchor set. We first show that the resulting
label-efficiency ceiling cannot be moved by any of four label-free encoder
adaptation families (test-time entropy minimization, low-rank continued
pretraining, cross-domain transfer, and pseudo-label feedback). We then show
that the ceiling *can* be moved by **anchor recycling**: the same anchors that
supervise the semantic layer are reused as supervision for a low-rank,
distillation-anchored update of the physical layer. The update is zero-initialized,
keeps BatchNorm statistics frozen, and re-enters the same measurement harness.

**Why the evidence is strong.** Every number in the paper is produced by one
shared measurement implementation and audited against per-seed evidence:

- Four tiers of graded reference quality (NTU60, PanAf500, NTU120, UCF101):
  recycling gains +10.20/+9.71/+20.12 pp, crossing two pre-registered verdict
  bands, 10/10 seeds each.
- Matched-budget supervised baselines: full fine-tuning (80.32%), BitFit
  (76.02%), and bottleneck adapters (79.74%) all trail recycling (85.16%).
- Both directions of taxonomy evolution: a 60→49 merge (+11.25 pp over
  semantic-only) and a 60→120 class addition (+5.20 pp overall, new classes
  +7.57 pp, old-class forgetting reduced from 10.90 to 8.06 pp), 10/10 seeds.
- Scale and capacity: the advantage survives a balanced 120-class taxonomy
  (+10.84 pp) and a 4× parameter range (16/32/64 channels:
  +10.20/+15.07/+8.84 pp), with the standard-width (6.36 M) encoder included.
- A two-sided envelope: an anchor-mass floor below which unaugmented
  recycling is inert, and a calibration-gate ceiling above which the semantic
  layer refuses the adapted features.

**Statements.** This manuscript is original, has not been published
previously, and is not under consideration elsewhere. All authors have read
and approved the submission and declare no conflicts of interest. The
measurement harness, configuration files, and all per-seed results are
released with the paper to allow full reproduction. [Optional: data are
publicly available from NTU, PanAf20K, and UCF101.]

We believe the paper offers the community both a practical mechanism for
label-efficient adaptation under changing taxonomies and an unusually
thorough negative-result map of the alternatives.

Sincerely,
[Corresponding author name, on behalf of all authors]
[Affiliation, email]

---

## 定制段（按刊选用）

### TMM（IEEE Transactions on Multimedia）— 推荐首选

> The paper fits TMM's scope on multimedia understanding and human-centered
> motion analysis: it addresses skeleton-based behavior recognition, a
> multimodal sensing modality widely used in surveillance, healthcare, and
> human–computer interaction, and it targets the label-scarce, evolving
> deployment settings that practical multimedia systems face. All evidence is
> measured under one shared protocol with ten seeds per configuration, and the
> paper releases its full per-seed results.

### PRL（Pattern Recognition Letters）

> As a Letter, the paper reports a focused mechanism — anchor recycling —
> with an unusually complete validation: four adaptation baselines that fail,
> two directions of taxonomy evolution, three encoder capacities, and a
> two-sided operating envelope, all under one shared ten-seed protocol. We
> believe this compact mechanism-plus-boundary study suits PRL's format.

### CVIU（Computer Vision and Image Understanding）

> The paper fits CVIU's interest in understanding-oriented vision systems:
> it couples a behavior-recognition architecture with a systematic study of
> what limits label efficiency under evolving taxonomies, including a
> negative-result map and an operating envelope, evaluated on human and
> non-human (great ape) behavior tiers with a released measurement harness.
