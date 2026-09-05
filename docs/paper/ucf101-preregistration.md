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

## Amendment (2026-09-05, protocol deviation + first-run bug, disclosed)

1. **Pretext architecture**: frozen text specified from-scratch ST-GCN+BC (100 ep). Executed pretext is a **joint-level MLP** (80 ep, frozen 256-d penultimate) — same architecture as P5-B for cross-dataset comparability with arbitrary joint counts. Retention-ratio protocol and (c)-absorbs-backbone logic unchanged; the weak (c) reference (23.11%) reflects the probe-quality pretext (PYSKL-lineage HRNet skeletons on UCF101 are known-weak).
2. **Script**: executed via `scripts/run_p5b_generic_retention.py` (generic V-joint version), not the named `run_p20_ucf101_retention.py`.
3. **First-run defect (fixed before results were read)**: arm-(b) scorer compared string predictions to integer labels (always False), recording b=0.0 spuriously. Fix (score in string domain) changed no arm, budget, seed, or decision rule; the re-run is the first valid readout.

**Result (2026-09-05, n=10000 clips, 101 classes, seeds 42/43/44)**: (c) full ref 23.11%; (a) 10% linear 14.04% (linear retention 60.75%); (b) PSD semantic pipeline 12.74/14.47/16.20 → mean 14.47%, **retention 62.62% → FAILS** (<85% line). PSD arm exceeds 10% linear by only +0.4pp. Honest report per the frozen rule: budget behavior narrows further; E9's claim is scoped to its measured domains. Cross-domain gradient (NTU60 90.6% → NTU120 89.2% → UCF101 62.6%) tracks the (c)-reference quality (74.3% → 54.3% → 23.1%), indicating pretext feature quality is the dominant factor in retention.

## 5. Reproduction

Executed: `scripts/run_p5b_generic_retention.py --pkl data/pyskl/ucf101_hrnet.pkl --name ucf101 --max 10000`; evidence `reports/p5b-ucf101-retention-2026-09-05.json`.
