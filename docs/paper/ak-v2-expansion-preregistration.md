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
