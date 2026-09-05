# Pre-registered protocol: label-alignment pseudo-label correction (P1 line)

> **Protocol ID**: PSD-ALIGN-PREREG-001 | **Registered**: 2026-09-05, frozen BEFORE any P1 experiment run (ADR 0007)
> **Purpose**: test whether a label-alignment force—constraining pseudo-labels to agree with the seed-anchored prototype geometry—rescues end-to-end low-resource recognition on the canine public-real tier, where R16 established the corrected protocol collapses to near-chance (warm spc2 9.8%±7.5, pool pseudo-label precision ≈0.11).
> **Iron rules**: all arms GT-free (no pool ground-truth in any training/gating/stopping path, per R16 correction); final head = seed truth + pool pseudo-labels; stopping = convergence or iteration budget (precision-drop disabled); decision rule frozen here and not revisitable after seeing results; any post-hoc interpretation appended as a dated amendment, never a silent edit.

## 1. Failure mechanism being targeted (from R16 artifacts, cited not assumed)

- r0 prototype path (seed-anchored class means): pool precision 0.30 at coverage 0.35.
- After the MLP head takes over assignment with 18 seeds: precision 0.20 → 0.17 while coverage inflates to 0.78 (head-margin scale ≠ prototype-calibrated τ*).
- GT-free per-round anchor-side recalibration (head_calib) does not rescue (8.9–12.5%).
- Diagnosis: the head drifts from the only label-aligned structure available (anchor prototypes); confirmation bias compounds.

## 2. Arms (all share R16-corrected protocol; differ ONLY in the gate/assignment mechanism)

| Arm | Mechanism (GT-free) |
|---|---|
| V0 control | R16 as-is: head-path assignment, standing-only consensus gate (inert on AK), frequency-aware margin τ_c |
| V1 anchor-consensus gate | proposal enters pool iff head top-1 == prototype-path top-1 AND head margin κ ≥ τ_c; prototypes re-estimated from seeds∪pool each round |
| V2 V1 + prior-matched quota | V1, then per pseudo-class cap: accept top-K_c by κ, K_c = max(1, round(0.35·|U|·π_c)), π_c = SEED class prior (frozen at r0, never updated) |
| V3 prototype-primary iteration | assignment always by prototype path (p_pred, calibrated p_margin); head trained but never assigns; prototypes re-estimated from seeds∪pool |

τ* selection, temperature calibration, frequency-aware margins, head config, budgets, splits, leakage guards: identical to `reports/r16-endtoend-pseudo-2026-09-05.json` (driver `scripts/run_r16_endtoend_pseudo.py`).

## 3. Primary endpoint and decision rule (frozen)

**Primary**: AK v1 (full12, 9 classes with samples, 197 clips), spc2 (18/141 ≈ 13%), seeds 42–51 (n=10), end-to-end val top-1 mean. Best variant chosen by mean top-1 computed once; ties broken by win-count vs V0.

- **ALIGNS**: best-variant mean ≥ 20.0% AND paired seed wins vs V0 ≥ 8/10.
- **PARTIAL**: mean in [15.0%, 20.0%) OR wins 5–7/10.
- **NULL**: mean < 15.0% (and not PARTIAL).

**Secondary (diagnostics only, never claims)**: per-round pool oracle precision (post-hoc, outside control path), macro-F1, coverage.

## 4. Consequences by verdict (frozen)

- **ALIGNS** → winner replicates on AK v2 (spc2, 10 seeds) and NTU (arm b, 3 seeds) as pre-registered replication layers; paper E7/E7b/E9 and abstract rewritten with new dated artifacts; any claim UPGRADE requires a fresh user ruling (C7 precedent).
- **PARTIAL** → reported as a bounded positive: mechanism works partially, tier still bottlenecked; paper keeps negative boundary + adds the alignment ablation to §5.
- **NULL** → negative boundary stands; V0–V3 become a §5 methodological ablation ("label-alignment force alone does not rescue this tier at 13%"); no rescue claim anywhere.

## 5. Reproduction

Implementation: `psd/training/tcl_selftrain.py` (`gate_mode` ∈ {standing, consensus_all, consensus_all_quota, proto_primary}; default "standing" preserves all prior behavior). Driver: `scripts/run_p15_align.py`. Evidence: `reports/p15-label-alignment-<date>.json` (+ paired tests).
