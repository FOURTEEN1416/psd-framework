# Pre-registered protocol: UCF101 as second independent cross-domain retention point (P5-A′)

> **Protocol ID**: PSD-UCF101-PREREG-001 | **Registered**: 2026-09-05, frozen BEFORE any run (ADR 0009)
> **Purpose**: after E9 (NTU60, 90.6% retention) and P5-B (NTU120), test the semantic pipeline's budget behavior on a genuinely independent third domain — UCF101 (101 action classes, YouTube-sourced real video, HRNet 2D skeletons from the PYSKL pre-processed release) — turning cross-domain evidence into three points across two skeleton modalities and three video sources.

## 1. Data (frozen)

`data/pyskl/ucf101_hrnet.pkl` (openmmlab CDN direct; 705 MB; 13,320 clips; 101 classes; HRNet 2D 17-joint skeletons pre-extracted; split1 train/test). Truly independent from NTU60/120: different video source (YouTube vs Kinect lab), different action space, different skeleton extractor (HRNet vs Kinect SDK).

## 2. Method (frozen — mirrors P5-B exactly)

- **Pretext**: from-scratch ST-GCN+BC, 100 epochs, on UCF101 train split (single GPU budget). Fidelity absorbed by the (c) reference arm.
- **Arms**: (c) 100% train linear head reference; (a) 10% stratified linear head; (b) 10% + PSD semantic pipeline (corrected protocol, precision_stop=False, final head on seed truth + pool pseudo).
- **Budget**: 10% class-stratified across 101 classes, seed 42 subset, selftrain seeds 42/43/44.

## 3. Decision rule (frozen)

- **CONFIRMS**: retention(b) ≥ 90% → third cross-domain point clears the same bar.
- **PARTIAL**: 85–90%.
- **FAILS**: <85% → honest report: budget behavior narrows further; E9's claim scoped to its measured domains.

## 4. Disclosures (frozen)

- From-scratch 100ep pretext (compute-limited); (c) arm absorbs backbone quality.
- UCF101 is a second human benchmark (after NTU60/120). Animal-domain cross-validation remains the target of P2′ and P7.

## 5. Reproduction

`scripts/run_p20_ucf101_retention.py`; evidence `reports/p20-ucf101-retention-<date>.json`.
