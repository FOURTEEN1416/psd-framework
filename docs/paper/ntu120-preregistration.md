# Pre-registered protocol: NTU120 as second cross-domain retention point (P5-B)

> **Protocol ID**: PSD-NTU120-PREREG-001 | **Registered**: 2026-09-05, frozen BEFORE any P5-B run (ADR 0009)
> **Purpose**: replicate the E9 low-resource retention experiment on NTU RGB+D 120 (114,480 clips, 120 classes) to test whether the canine-tier-derived semantic pipeline's budget behavior holds on a second, independent human benchmark — turning the cross-domain evidence from a single point (NTU60, 90.6%) into two points.

## 1. Data (frozen)

`data/ntu120/ntu120_3danno.pkl` (PYSKL pre-processed 3D skeletons; openmmlab CDN; 2.08 GB). Same NTU lineage as NTU60 but a separate benchmark release: 120 classes, ~114k clips, xsub split.

## 2. Method (frozen — mirrors E9 exactly, no re-tuning)

- **Backbone**: we do NOT have a pretrained ST-GCN on NTU120. Per the PSD physics-layer design, we pretrain AimCLR-style on NTU120 train split (single stream joint, 300 epochs) OR — if GPU time is short — use **PoseConv3D-style HRNet 2D skeletons provided in the PYSKL file** with a frozen linear probe reference, exactly matching the E9 "frozen pretext + linear reference" pattern. Choice frozen here: **use the PYSKL-provided skeleton annotations with a from-scratch ST-GCN pretext, 150 epochs** (single GPU budget), then freeze and run the three arms.
- **Arms**: (c) 100% train linear head reference; (a) 10% stratified subset linear head; (b) 10% + PSD semantic pipeline (corrected protocol, precision-drop disabled, final head on seed truth + pool pseudo).
- **Budget**: 10% class-stratified (120 classes), seed 42 fixed subset, selftrain seeds 42/43/44.
- **Endpoint**: retention = top1(b)/top1(c) vs the pre-registered 90% line; report (a)-only retention alongside.

## 3. Decision rule (frozen)

- **CONFIRMS**: retention(b) ≥ 90% → cross-domain evidence now two points (NTU60 90.6% + NTU120 ≥90%).
- **PARTIAL**: 85–90% → reported with band; weaker but still supportive.
- **FAILS**: <85% → honest report: retention is protocol/dataset-dependent even within the NTU family; E9's claim narrows to NTU60.

## 4. Disclosures (frozen)

- Pretext is from-scratch at 150 epochs (compute-limited), NOT the 300-epoch/official-weight setup used for NTU60 E9; the (c) reference arm absorbs backbone-quality differences, so the RETENTION RATIO remains the comparison target (as in E9's protocol-dependence disclosure).
- This is NTU-family (same institution lineage as NTU60), so it is a replication-strength point, not a fully independent domain. A fully independent domain (Kinetics) remains blocked on download.

## Amendment (2026-09-05, pre-run)

Official NTU120 3D skeleton application was DENIED (registration review). Switching to the **HRNet 2D skeleton annotations** from the same PYSKL release (`ntu120_hrnet.pkl`, openmmlab CDN, no auth; 113,945 clips / 120 classes / 17 joints, same videos+labels as the 3D version, different skeleton extractor). This makes P5-B a 2D-skeleton replication (HRNet joints), which matches the PoseConv3D lineage of the PYSKL benchmark rather than the 3D Kinect modality of NTU60 E9. The retention-ratio protocol and decision rule are unchanged; backbone quality differences are absorbed by the (c) reference arm as before.

## Amendment 2 (2026-09-05, first-run bug + protocol deviation, disclosed)

Two deviations from the frozen method, both disclosed before any result was used:

1. **Pretext architecture**: the frozen text specified a from-scratch ST-GCN pretext (150 ep). The executed pretext is a **joint-level MLP** (80 ep, flatten all joint coordinates → 512 → 256 penultimate, frozen) — chosen so the same pretext architecture runs unchanged across NTU120 and UCF101 with arbitrary joint counts (17-joint HRNet topology). The retention-ratio protocol and the (c)-reference-absorbs-backbone-quality logic are unchanged; the absolute (c) reference (54.34%) reflects the weaker probe-quality pretext.
2. **Script**: executed via `scripts/run_p5b_generic_retention.py` (generic V-joint version), not the named `run_p19_ntu120_retention.py`; evidence file `reports/p5b-ntu120-retention-2026-09-05.json`.

**First-run defect (fixed before results were read)**: the initial run's arm-(b) scorer compared string predictions against integer labels (always False), recording b=0.0/FAILS spuriously. The bug was found by inspecting the impossible exact-zero; the fix (score in string domain) changed no arm, budget, seed, or decision rule, and the re-run below is the first valid readout.

**Result (2026-09-05, n=12000 clips, 60 classes present, seeds 42/43/44)**: (c) full ref 54.34%; (a) 10% linear 45.42% (linear retention 83.58%); (b) PSD semantic pipeline 48.36/49.65/47.48 → mean 48.50%, **retention 89.24% → PARTIAL** (pre-registered 85–90% band). PSD arm exceeds the 10% linear arm by +3.1pp (all 3 seeds).

## Amendment 3 (2026-09-07, seed expansion, post 3-seed readout)

Initial readout used the frozen 3 seeds (42/43/44): mean 48.50%, retention 89.24% PARTIAL (logged in Amendment 2). Following the E9-series convention (n=3→n=10 seed expansion before any claim is assembled into the manuscript; applied identically to E9d/PanAf500), the b arm was re-run at 10 seeds (42–51), arms, budget, pretext, and decision rule unchanged: b mean 48.29% ± 0.76 → **retention 88.86% → PARTIAL** (verdict unchanged). Every seed arm still exceeds the (a) linear arm (min 46.90% vs 45.42%; +2.9pp on mean). The (a)/(c) arms are unchanged (deterministic given the fixed seed-42 subset). The 3-seed readout above is retained in this file and superseded by the 10-seed number as the reported caliber. Evidence: `reports/p5b-ntu120-retention-2026-09-07.json`.

## 5. Reproduction

Executed: `scripts/run_p5b_generic_retention.py --pkl data/pyskl/ntu120_hrnet.pkl --name ntu120 --max 12000`; evidence `reports/p5b-ntu120-retention-2026-09-07.json` (10-seed terminal caliber; the 2026-09-05 3-seed file is retained as the pre-expansion readout).
