# Pre-registered protocol: seed-budget curve localization (P3)

> **Protocol ID**: PSD-BUDGET-PREREG-001 | **Registered**: 2026-09-05, frozen BEFORE any P3 run (ADR 0008)
> **Nature**: DESCRIPTIVE characterization — no pass/fail decision rule; this locates where the canine-tier pipeline becomes usable, it does NOT by itself license a low-resource claim.

## 1. Purpose

P1/P2 test whether the mechanism can be fixed at 13% labels. P3 answers the orthogonal, decision-relevant question: at what annotation budget does the corrected pipeline (with the best P1 gate) actually become usable on AK v1? This turns the negative boundary into a quantitative knee-point characterization ("the pipeline needs ≥X% labels on this tier"), which is itself a publishable low-resource finding and directly informs the real-K9 pilot's budget design.

## 2. Arms (frozen)

V0 (control) and V2 (P1-best gate) × spc ∈ {2,3,4,6,8,12} × seeds 42–51 (n=10), AK v1, warm features, R16-corrected protocol (final head = seed truth + pool pseudo; oracle stopping off). spc3 ≈ 19% labels (within the C6 ≤20% envelope); spc12 ≈ 76%.

## 3. Reported quantities (frozen)

Per (arm,spc): mean±std val top-1, macro-F1, retention vs the 33.93% full-supervision reference, pool size, pool oracle-precision diagnostic (post-hoc only). **Knee** = smallest spc whose V2 mean top-1 ≥ 24.0% (70% retention), by linear interpolation between bracketing points; reported with the bracket, not a fabricated precise value.

## 4. Claim discipline (frozen)

- P3 is exploratory/descriptive. If the knee lands at ≤20% labels (spc≤3), reviving a positive canine low-resource claim requires a SEPARATE confirmatory pre-registration on fresh seeds and a user ruling (exploration and confirmation are separated; no same-data find-and-confirm).
- If the knee is >20%, the paper states the measured usability boundary honestly; no rescue implied.

## 5. Reproduction

`scripts/run_p17_budgetcurve.py`; evidence `reports/p17-budget-curve-<date>.json`.
