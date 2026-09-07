# Pre-registered protocol: PSD low-resource retention on NTU60 (human-domain reference)

> **Protocol ID**: PSD-NTU-PREREG-001 | **Registered**: 2026-09-04, before any NTU low-resource experiment
> **Purpose**: test whether the paper's central low-resource claim (semantic warm-start + pseudo-label self-training retains most of full-supervision accuracy at small budgets) generalizes to the human skeleton domain, where a published budget reference exists (TCL: 82.7% at 10% labels vs 88.6% full = 93.3% retention).

## 1. Comparison axis (frozen)

**Retention ratio, not absolute accuracy.** Our protocol is a frozen-pretext linear/self-trained head; TCL's is full fine-tuning of their pipeline. Absolute numbers are not comparable across protocols and we never claim they are. The claim under test is behavioral: *how much of its own full-budget accuracy does PSD keep when the label budget shrinks to 10%?* — measured on the same benchmark TCL reports, so the retention ratios are directly comparable with the same caveat.

## 2. Setup (frozen)

| Element | Specification |
|---|---|
| Benchmark | NTU RGB+D 60, xsub split, joint stream |
| Backbone | epoch300 joint pretext checkpoint (`runs/ntu_phaseB/joint_pretext/epoch300_model.pt`), frozen — the same weights whose full-budget linear eval produced the paper's 74.30% equivalence row |
| Features | official AimCLR `Feeder_single` (no augmentation, mmap) → `encoder_q` penultimate 256-d backbone features, dumped once for all 40,091 train + 16,487 val clips |
| Budget | 10% of train clips, stratified per class (60 classes), fixed by seed 42; full-budget arm uses 100% |
| Arms | (a) linear head on 10% only; (b) PSD semantic pipeline (anchor-guided clustering + iterated confidence-filtered pseudo-labeling, same `run_selftrain` as E7) on 10%; (c) linear head on 100% (reference) |
| Seeds | 3 for stochastic arms; deterministic arms reported once |
| Leakage | pseudo-label pool restricted to train clips only; val never enters the pool |

## 3. Endpoints and decision rule (frozen)

- **Primary**: retention = top1(b) / top1(c).
  - ≥ 90% → low-resource claim generalizes to the human domain (reported as cross-domain evidence).
  - 85–90% → partial generalization, reported with the band.
  - < 85% → reported as-is as an animal-domain-specific effect boundary (a limitation, not deleted).
- **Secondary**: arm (a) vs (b) isolates the pseudo-label iteration's contribution at human scale; comparison of our retention vs TCL's 93.3% stated with the protocol caveat.

## 4. Disclosure requirements

TCL's 82.7/88.6 pair comes from their fine-tuned pipeline; our reference arm is a frozen probe (74.30% full). We cite TCL's retention as a published reference point on the same benchmark, never as a head-to-head accuracy win. All three arms, seeds, and the 10% subset are released as a manifest.

## 5. Reproduction

Feature dump: `scripts/run_p14_ntu_featuredump.py` → `runs/ntu_lowres/features_joint_ep300.npz`. Protocol: `scripts/run_p14_ntu_lowres.py` → `reports/p14-ntu-lowres-<date>.json` + `.md`.

## 6. Amendment (2026-09-04, post-registration, pre-submission)

R8 citation audit falsified the TCL reference used in §1: the cited paper (arXiv 2102.02751) contains no NTU60 experiment and no 82.7/88.6 figures (full-text PDF search: NTU 0 hits). The TCL comparison is therefore **removed**; the primary endpoint stands on its own pre-registered 90% line only. Additional disclosures added post hoc: the final head uses StandardScaler + logistic regression (tol 1e-3) and GPU heads (scale adaptations); the 10% subset is a single stratified draw; retention ratios are protocol-dependent (frozen-probe curves are flatter by construction). None of these changes the arms, budget, seeds, or decision rule.

## 7. Amendment (2026-09-05, post-registration, pre-submission — R16 protocol correction)

R16 adversarial review found that the executed arm (b) deviated from the low-resource semantics of §2/§3: the reported classifier was retrained on the TRUE labels of the pseudo-labeled pool clips (≈38.4k of 40,091 training clips ≈ 96% label consumption), and the iteration's precision-drop stopping consumed official training labels. Corrected protocol: the final head consumes seed ground-truth labels plus pseudo-labels for pool clips; the precision-drop rule is disabled (no rule-derived consensus reference exists on this benchmark); pool precision vs official labels (≈0.69) is a post-hoc diagnostic outside the control path. Corrected result: top1(b) = 67.5% ± 0.15 (three seeds), retention = 90.6% of the unchanged supervised reference 74.45% — the pre-registered ≥90% GENERALIZES verdict still holds; the pseudo-iteration gain over arm (a) is +1.4pp (was +8.1pp under the erroneous head). Arms (a)/(c) are pure supervised and unaffected. Evidence: `reports/r16-ntu-pseudo-2026-09-05.json`; driver `scripts/run_r16_ntu_pseudo.py`. The §6 amendment remains valid.

## 8. Amendment (2026-09-07, seed expansion, post 3-seed readout — E9-series convention)

Initial readout used the frozen 3 seeds (42/43/44): top1(b) = 67.5% ± 0.15, retention 90.6% GENERALIZES (logged in §7). Following the E9-series convention (n=3→n=10 seed expansion; applied identically to E9b/E9c/E9d), the b arm was re-run at 10 seeds (42–51), arms, budget, 10% subset (seed 42), and decision rule unchanged: top1(b) = 67.53% ± 0.24 → **retention 90.7% → GENERALIZES** (verdict unchanged; margin over the pre-registered 90% line widens slightly from +0.6pp to +0.7pp). The (a)/(c) arms are deterministic and re-verified bit-identical to the 09-05 artifact (74.45%/66.05%) inside the new self-contained artifact. The 3-seed readout above is retained and superseded by the 10-seed number as the reported caliber. Evidence: `reports/r16-ntu-pseudo-10seed-2026-09-07.json`; driver `scripts/run_r16_ntu_pseudo_10seed.py` (separate script; the frozen `run_r16_ntu_pseudo.py` is unmodified).
