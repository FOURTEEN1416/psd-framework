# Pre-registered protocol: PSD working-dog (real-K9 tier) pilot

> **Protocol ID**: PSD-K9-PREREG-001 | **Registered**: 2026-09-04 (before any pilot data collection)
> **Status**: AWAITING DATA AUTHORIZATION — no working-dog footage with behavior labels exists in either repository as of registration date (see §1).
> **Lock discipline**: this document is committed to the public review repository at the registered commit; any post-hoc deviation must be reported as a deviation, never silently absorbed.

## 1. Why this protocol is registered before the experiment

The paper's three-tier protocol (synthetic / public-real / real-K9) leaves the real-K9 column blank: no public dataset contains labeled working-dog training footage, and the product-line risk register (k9-training-system ADR 0008 v1.7, 2026-08-19) formally retired the only candidate set (5 public YouTube dog videos, 682 clips) after confirming it carries **zero behavior annotations**. Rather than extrapolate, we pre-register the pilot so that target-domain evidence, when it arrives, cannot be cherry-picked post hoc.

## 2. Hypotheses

- **H1 (primary)**: the PSD pipeline, warm-started exactly as in the public-real tier (frozen physics backbone + semantic-layer self-training), reaches ≥ 60% top-1 on a 6-class working-dog behavior set (sit / down / stay / heel-track / watch / retrieve) at ≤ 5 labeled clips per class, versus ≤ 40% for a scratch-initialized control under the identical budget.
- **H0**: no difference beyond the seed-noise band.
- **H2 (secondary, decoupling claim)**: a taxonomy transition (adding one behavior) is absorbed by the semantic layer alone at ≥ 3× lower wall-clock cost than full-pipeline retraining, measured on real footage.

## 3. Data

| Item | Specification |
|---|---|
| Source | Working-dog training footage from the partner organization (deployment uploads; product-line pipeline, read-only import) |
| Volume target | ≥ 30 clips per class × 6 classes (T=30, 24-joint dog topology via the same YOLO11-pose dog-pose extractor as the public-real tier) |
| Exclusion rule | clips with mean keypoint confidence < 0.20 or no detected dog in > 50% of frames (identical to public-real gates) |
| Split | by session (dog × day), never by clip: all clips from one dog-day go to one split; ≥ 8 distinct dogs |

## 4. Annotation protocol (the gate that killed the retired set)

1. Two independent annotators, Label Studio, behavior class per clip, no adjudication during first pass.
2. **Agreement gate**: Cohen's κ ≥ 0.60 required before any model is run on the labels; below gate → taxonomy revision, re-annotate, re-gate. (The retired set failed at step zero: no annotations existed.)
3. Disagreements adjudicated by a third expert; adjudicated label is final and recorded with its provenance.
4. Annotation codebook, κ values, and disagreement counts are published with results.

## 5. Arms and protocol

Identical to the public-real end-to-end protocol (p07 driver, unchanged code): seeds (spc ∈ {2,5}) → anchor-guided clustering → iterated pseudo-labeling → seeds+pool linear head → session-split validation. Arms: (a) PSD warm-start; (b) scratch control; (c) generic-SSL control (AimCLR, per p08). Three seeds each; deterministic leakage guards (validation sessions never enter the pseudo-label pool).

## 6. Endpoints and analysis

- **Primary**: top-1 on held-out sessions, PSD-warm vs. scratch, paired per seed; report mean ± std and per-class accuracies (aggregate-only reporting prohibited, per paper L7).
- **Secondary**: macro-F1; H2 wall-clock ratio.
- **Multiplicity**: single primary comparison, no family-wise correction needed; secondary endpoints labeled as such.
- **Decision rule**: H1 accepted iff paired mean difference ≥ +15 pp AND every seed pair positive; otherwise reported as a pre-registered negative.

## 7. Ethics and release

Footage rights remain with the partner organization; consent for research use recorded per session. **No raw footage, no derived skeletons, and no per-dog identifiers are redistributed**; the public repository releases the protocol, the labeling codebook, the driver scripts, and aggregate result JSONs only.

## 8. Deviations

Any change to §§2–6 after data collection begins is reported in the paper as a named deviation with its rationale. The registration date and this document's commit hash are the audit anchor.
