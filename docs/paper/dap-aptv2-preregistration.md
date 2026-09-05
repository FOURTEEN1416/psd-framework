# Pre-registered protocol: APTv2 domain-adaptive pretraining backbone (P2)

> **Protocol ID**: PSD-DAP-PREREG-001 | **Registered**: 2026-09-05, frozen BEFORE any P2 run (ADR 0008)
> **Purpose**: test whether a stronger physics-layer backbone—domain-adaptive continued pretraining (DAP) on unlabeled APTv2 canid skeletons—lifts the canine-tier end-to-end low-resource accuracy above the P1-improvable ceiling, since P1 (PSD-ALIGN-PREREG-001) localized the binding constraint to the seed-anchored feature geometry (r0 prototype precision ≈0.30), not the gating rule.
> **Prior (cited, not ignored)**: L9 established that AdaBN-style second-order-statistics adaptation from the same APTv2 pool did NOT yield aggregate gain ("conversion, not supply"). P2 is a strictly stronger, representation-level intervention (gradient-bearing contrastive pretraining), not a repeat of L9; if P2 also fails it corroborates that APTv2→AK transfer is bottlenecked by topology/appearance mismatch, not adaptation strength.

## 1. Data (GT-free)

503 unlabeled APTv2 canid skeleton clips (`runs/data_campaign/aptv2/sequences/canidae/*.pkl`, shape (30,24,3), uniform_T30 resampled, harmonized keypoints). No labels consumed. Held-out: none needed (unsupervised); AK val stays isolated as in all prior arms.

## 2. Method (frozen)

Backbone: ST-GCN+BC initialized from Y_CKPT (`runs/p05_stgcn_bc_full/best.pt`). Objective: two-view InfoNCE (temperature 0.1, queue 1024) over augmentations = random temporal crop (±25%) + joint jitter (σ=0.02) + uniform scale (0.9–1.1); AdamW lr 1e-4, 60 epochs, batch 32, seed 42 (single DAP run; downstream seeds vary). Hyperparameters fixed here, not tuned on AK.

## 3. Arms and endpoint (frozen)

After DAP, re-extract AK v1 features and rerun the R16-corrected protocol with the P1-best V2 gate, spc2, seeds 42–51 (n=10). Compare to P1-V2 baseline (14.11%±6.65).
- **Fidelity self-check (run first)**: full-supervision linear reference on DAP features; if < 30% (vs 33.93% pre-DAP), flag catastrophic forgetting and interpret any downstream delta as confounded.
- **Primary endpoint**: mean end-to-end val top-1, DAP+V2 vs P1-V2 (same seeds, paired).

## 4. Decision rule (frozen; same bar as P1 — no goalpost move)

- **RESCUE**: DAP+V2 mean ≥ 20.0% AND paired wins vs P1-V2 ≥ 8/10.
- **PARTIAL**: mean in [15.0%, 20.0%).
- **NULL**: mean < 15.0%.

## 5. Consequences (frozen)

- RESCUE → confirmatory replication on AK v2 + NTU under a NEW pre-registration; paper E7/abstract rewritten with a positive canine claim (user ruling required for upgrade, C7 precedent).
- PARTIAL/NULL → negative boundary stands; P2 becomes a §5 ablation ("stronger backbone via APTv2 DAP does not cross the rescue bar; the constraint is intrinsic to the AK self-extraction tier"), reinforcing the tier-dependent thesis.

## 6. Reproduction

`psd/training/dap_infonce.py` (new module), `scripts/run_p16_dap.py`; evidence `reports/p16-dap-aptv2-<date>.json`.
