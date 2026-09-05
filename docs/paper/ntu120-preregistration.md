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

## 5. Reproduction

`scripts/run_p19_ntu120_retention.py`; evidence `reports/p19-ntu120-retention-<date>.json`.
