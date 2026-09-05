# Pre-registered protocol: AK public-real tier v2 multi-segment expansion

> **Protocol ID**: PSD-AKV2-PREREG-001 | **Registered**: 2026-09-04, before any v2 dataset build or experiment
> **Relationship to v1**: v1 (197 clips, one per video) remains the paper's primary public-real tier. v2 is a pre-registered **replication/robustness layer** — no v1 number is replaced.

## 1. Motivation (stated before seeing any v2 result)

The paper attributes the public-real accuracy ceiling (~34%) to the self-extraction data bottleneck (L7): v1 extracts exactly **one clip per video** (`seen.add(vid)` in the v1 selector), while canine AK videos carry median 80 / p75 139 / max 447 labeled frames. If the ceiling is data-limited, a 1.5–2× larger training pool should raise the full-supervision arm measurably; if it is task-intrinsic (label noise, viewpoint, 20/24 channels), the ceiling should stay flat. Either outcome is informative; both will be reported.

## 2. Dataset design (frozen)

| Element | Specification |
|---|---|
| Eligibility | canine species ∩ 12-class PSD mapping pool (identical to v1) |
| Segmentation | per video, K = min(4, max(1, n_frames // 40)) contiguous equal segments, each ≥ 40 frames |
| Segment label | pool class with highest frame coverage; **gate: coverage ≥ 0.80** (frame label = set of PSD classes mapped from the comma-separated AK multi-label) |
| Clip construction | 30 frames uniform-sampled **within the segment**, same YOLO11s-pose dog-pose weights (`runs/public_real_yolo_dogpose/train/weights/best.pt`), same `assemble_clip` (interpolation, dead-joint hard mask), same quality gates (mean vis ≥ 0.20, conf ≥ 0.30) |
| Splits | video-level, AK official train/val csv membership — unchanged from v1; all segments of one video land in one split |
| Output | `runs/public_real_dataset/full12v2_T30.pkl` + manifest + quality funnel JSON |

## 3. Endpoints (all reported regardless of direction)

- **EP1 — yield**: v2 clip count and per-class distribution vs v1 (expect minority-class growth: sit/retrieve/bite/apprehend had 7–8 clips in v1).
- **EP2 — replication**: E7 arms (warm / scratch / AimCLR) at spc2, ten seeds, identical p07/p10 protocol on v2.
- **EP3 — ceiling test**: full-supervision arm on v2 vs v1's 33.93%.

## 4. Decision rule (frozen)

- **Data-bottleneck hypothesis confirmed** iff v2 full-supervision top-1 ≥ v1 + 3.0 pp.
- **Task-intrinsic ceiling** iff v2 full-supervision top-1 within ±3.0 pp of v1.
- Below v1 − 3.0 pp: report as v2 degradation with funnel analysis (segment length / label noise candidates).
- No arm, seed, or gate may be changed after the build; any deviation is named in the paper.

## 5. Leakage and disclosure

The warm-start backbone (Y_CKPT) was supervised on v1 training-split videos; v2 adds new **segments of the same videos** to its training pool — inherent to a task-pretrained warm start and disclosed in the paper (val videos remain disjoint from all training videos, so validation is clean). The AimCLR arm is unaffected (pretrained on InterPet4D mocap).

## 6. Reproduction

Builder: `scripts/run_p11_ak_v2_build.py` (this protocol's §2 implemented verbatim). Replication driver: `scripts/run_p12_ak_v2_replicate.py` (reuses p07/p08/p10 protocol functions). Evidence: `reports/p11-akv2-build-*.json`, `reports/p12-akv2-replication-*.json` + `.md`.

## 7. Amendment (2026-09-04, post-registration)

R8 adversarial review identified that the EP3 raw-top-1 threshold is confounded by the class-space change (v1 12-class chance 8.33% vs v2 8-class chance 12.5%; the +4.17pp chance shift exceeds the measured +3.57pp). A **same-space control** is added: re-scoring the v1 full-supervision arm on the eight classes v2 retains yields 25.96%, so the same-space rise is +11.54pp (above-chance 13.5→25.0pp; above-majority −4.8→+3.1pp). The data-bottleneck attribution is supported on top-1 under this control; macro-F1 moves oppositely (14.7→7.8, driven by v2's n=2 sit class) and is reported alongside. The original §4 rule text is preserved above unaltered; this amendment supersedes its interpretation.

## 8. Amendment (2026-09-05, post-registration, pre-submission — R16 protocol correction)

R16 adversarial review found that the executed EP2 low-budget arms trained the reported head on TRUE labels of the pseudo-labeled pool (protocol error; see the parallel NTU amendment §7). Corrected re-run (final head = seed truth + pool pseudo-labels; precision-drop stopping disabled): at 2 clips/class the warm arm reaches 13.1%±5.3, generic SSL 17.6%±4.3, scratch 13.8%±5.6 against 12.5% chance — near chance, as on v1; the previously reported 88.6% retention and warm-over-SSL widening are superseded (artifacts of the erroneous head). EP3 is unaffected in substance: its arms are pure supervised; the raw +3.57pp endpoint is additionally disclosed as within solver-path noise (re-encoding labels shifts the v2 full reference by 2.1pp, 37.50→35.42), so the data-bottleneck attribution rests on the §7 same-space control (+11.54pp). Evidence: `reports/r16-endtoend-pseudo-2026-09-05.json`; driver `scripts/run_r16_endtoend_pseudo.py`.
