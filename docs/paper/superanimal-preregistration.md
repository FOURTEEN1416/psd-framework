# Pre-registered protocol: SuperAnimal-Quadruped skeleton re-extraction (P2′)

> **Protocol ID**: PSD-SA-PREREG-001 | **Registered**: 2026-09-05, frozen BEFORE any P2′ run (ADR 0009)
> **Purpose**: test whether replacing the YOLO11-pose skeleton extractor with SuperAnimal-Quadruped (DLC Model Zoo; Ye et al., Nature Comms 2024; pretrained on 8.1K frames / 23 species) raises the AK canine tier's ceiling, since P1/P2/P3 triangulated that the binding constraint is data quality (skeleton extraction + label alignment).

## 1. Hypothesis

If the ceiling (full-supervision reference 33.93%) is skeleton-quality-limited, a stronger extractor will raise it and low-budget end-to-end accuracy will follow. If the ceiling is label-quality-limited (not skeleton), the stronger extractor will NOT move it—confirming that the constraint is intrinsic to the label pipeline, not the skeleton extractor.

## 2. Method (frozen)

- **Extractor**: SuperAnimal-Quadruped top-down (DLC Model Zoo, HF `mwmathis/DeepLabCutzoo-SuperAnimal-Quadruped`), 24-keypoint output mapped to the paper's 24-slot topology (same slot assignment as the current pipeline; the 4 dead slots masked as in L8).
- **Input**: the same 211 AK canine videos (`k9-training-system/data/animal_kingdom/action_recognition/dataset/video/`), same self-extraction clip assembly (YOLO-dog filter → clip boundary → resample T=30).
- **Confidence**: SuperAnimal's per-keypoint likelihood replaces YOLO-pose confidence; same NaN-guard downstream.
- **Downstream**: identical E7 corrected protocol (final head = seed truth + pool pseudo; precision-drop disabled; 10 seeds; warm + aimclr + scratch arms; spc2).

## 3. Primary endpoint (frozen)

Full-supervision linear-probe top-1 on the re-extracted AK v1 val split (vs.\ 33.93% baseline).
- **SKELETON_BOTTLENECK**: full-supervision reference rises by ≥3.0 pp → skeleton quality was a ceiling; proceed to low-budget rerun.
- **LABEL_BOTTLENECK**: full-supervision reference moves < 3.0 pp → skeleton quality is NOT the constraint; the label pipeline is. Do NOT proceed to low-budget rerun (report as confirmation).

## 4. Consequences

- SKELETON_BOTTLENECK → run low-budget arms (10 seeds); if low-budget accuracy also improves, the paper gains a positive canine result; if it improves full-supervision but not low-budget, report the decomposed finding.
- LABEL_BOTTLENECK → strengthen the negative boundary: "replacing the skeleton extractor with a stronger pretrained model did not lift the ceiling" is a powerful additional line of evidence that the constraint is the label pipeline, not the skeleton quality.
- Either way, evidence goes to §5 (or a new §6 if a positive result emerges).

## 5. Reproduction

`scripts/run_p18_superanimal_extract.py` (extraction), `scripts/run_p18_superanimal_e7.py` (downstream); evidence `reports/p18-superanimal-<date>.json`.
