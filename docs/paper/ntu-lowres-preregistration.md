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
