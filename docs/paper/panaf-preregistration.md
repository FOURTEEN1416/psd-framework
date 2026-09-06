# Pre-registered protocol: PanAf500 as first animal-domain public-benchmark retention point (P7)

> **Protocol ID**: PSD-PANAF-PREREG-001 | **Registered**: 2026-09-07, frozen BEFORE any P7 retention run (ADR 0009)
> **Purpose**: after three human-domain points (NTU60 90.6% / NTU120 89.2% / UCF101 62.6%), test the PSD semantic pipeline's budget behavior on the **first animal-domain public benchmark** — PanAf500 (great-ape natural behavior, in-the-wild video, official splits). This is the cross-species evidence the paper's motivation rewrite (P4 option 2) points to: no public working-dog behavior dataset exists, but a public animal benchmark can still test whether the pipeline's budget retention extends beyond the human domain.

## 1. Data (frozen)

`runs/p7_asbar/panaf500_T30.pkl` — 500 clips (official splits train 400 / val 25 / test 75), keypoints (T30, 24, 3) in PSD24 slot layout, 9 behavior classes. Skeletons extracted by the YOLO11x-pose ape model (AP-10K chimpanzee/gorilla/orangutan fine-tune, `p7_ape_pose/weights/best.pt`), bbox-cropped inference on PanAf500 videos, AP17→PSD24 mapping (17 slots mapped / 7 dead slots zeroed, L8-consistent). Extraction quality released: det_rate_mean 0.821, mean_conf 0.588 (`reports/p7-extract-quality.json`). Class distribution is long-tailed: walking 183 / sitting 161 / standing 96 / hanging 28 / climbing_up 13 / running 7 / camera_interaction 6 / sitting_on_back 3 / climbing_down 3.

## 2. Method (frozen — mirrors E9b/E9c exactly, no re-tuning)

- **Pretext**: joint-level MLP over all 24 joint coordinates (80 ep, frozen 256-d penultimate), trained on the official train split only — same architecture as P5-B/C for cross-benchmark comparability; backbone quality absorbed by the (c) reference arm.
- **Arms**: (c) 100% train linear head reference; (a) 10% class-stratified linear head; (b) 10% + PSD semantic pipeline (corrected protocol, precision_stop=False, final head on seed truth + pool pseudo).
- **Budget**: 10% of the 400 train clips, stratified per class (minority classes with ≤10 clips contribute 1 seed clip each — long-tail disclosed, no class merging); selftrain seeds 42/43/44.
- **Endpoint**: retention = top1(b)/top1(c) on the **val split** (25 clips), same as E9 series. The test split (75 clips) is held out of the decision rule; the final model's test accuracy is reported once as a non-decision secondary number.

## 3. Decision rule (frozen)

- **CONFIRMS**: retention(b) ≥ 90% → cross-domain evidence now spans the animal kingdom (human ×3 + primate ×1).
- **PARTIAL**: 85–90% → reported with band.
- **FAILS**: <85% → honest report: budget behavior does not extend to this animal tier; the claim remains scoped to its measured domains. **Either outcome is reported regardless of direction — no result-driven protocol modification.**

## 4. Disclosures (frozen)

- Skeletons are model-extracted (no ground-truth pose on PanAf), so extraction noise is inside the loop for ALL arms equally; det-rate 0.821/0.047 min spread is released with the artifact.
- The 9-class space is long-tailed (min class n=3); per-class seed coverage at 10% is 1 clip for 5 of 9 classes. This is disclosed as part of the tier's difficulty, not fixed post hoc.
- Pretext is from-scratch at 80 ep (compute-limited), consistent with E9b/E9c; (c) arm absorbs backbone quality.
- PanAf500 behavior labels are frame-level annotations majority-voted to clip level (video-level coarse granularity — the SAME label-granularity regime the paper diagnoses on AK, disclosed, not fixed).

## 5. Reproduction

Driver: `scripts/run_p23_panaf_retention.py`; evidence: `reports/p23-panaf-retention-<date>.json`.
