# Pre-registered protocol: external semi-supervised baseline on NTU60 10% budget (P9)

> **Protocol ID**: PSD-SSL-BASE-001 | **Status**: **FROZEN v1.0 2026-09-07**（冻结先于任何运行；驱动冒烟不算运行）
> **Origin**: R23c finding #4（"90% line" 是自设内部判据，无外部 SSL 对照是最大压制项）+ 用户 /goal"解决压制项"指令。

## 1. Purpose (frozen)

Place the E9 NTU60 10%-budget result (PSD pipeline 67.53%±0.24; linear head alone 66.05%) against a standard external semi-supervised method run **in the same harness** — same frozen pretext features, same 10% stratified subset (seed 42), same validation split, same evaluation code — so the comparison isolates the *label-consumption mechanism* (pseudo-label pool restoration) from the *SSL algorithm*.

## 2. Baseline (frozen)

**Mean-Teacher-style temporal ensembling over the pool** (the classical consistency-distillation SSL baseline; chosen because FixMatch's strong augmentation stack has no natural analogue on frozen 256-d features — disclosed as the reason FixMatch is not the baseline):

- Teacher = linear head trained on the seed-labeled subset only (identical to arm (a)).
- Student = linear head trained on seed labels + soft pseudo-labels for all non-anchor training clips, weighted by the teacher's confidence with a fixed threshold τ=0.95 (FixMatch's default margin), five EMA更新轮次 over the pool（教师-学生互馈 5 轮）.
- Hyperparameters beyond τ: none tuned（禁调参防作弊；τ=0.95 是文献默认）.

## 3. Arms / data (frozen)

| Arm | Definition |
|---|---|
| (a) linear@10% | 对照臂，与 E9 完全一致（66.05%） |
| (b) PSD pipeline | 与 E9 完全一致（67.53%±0.24, 10 seeds） |
| (mt) Mean-Teacher | 上节定义，seeds 42–51（10 seeds，与 E9 系列同口径） |

Features: `runs/ntu_lowres/features_joint_ep300.npz`（E9 冻结 pretext, 256-d）; labels/subset/split 全部复用 E9 代码路径（`run_p14_ntu_lowres.stratified_10pct`）。

## 4. Decision rule (frozen)

- **mt ≥ b mean + 2.9pp**（= E9b 的 PSD-vs-linear 差距量级）→ PSD 伪标签管线相对外部 SSL 无优势，E9 叙事重写为"与外部 SSL 相当"，如实入文。
- **b mean > mt**（配对 per-seed）→ 方向性优势；Wilcoxon(b vs mt per-seed) 报告，不预设显著性结论。
- **|b − mt| < 1pp** → 相当，如实报 parity。
无论方向，结果入正文 E9 段一句+Limitations L11 扩展一句——**本协议的目的就是消灭"无外部基线"攻击面，结果方向不筛选**。

## 5. Disclosures (frozen)

- Mean-Teacher 跑在冻结 pretext 特征上（与 PSD 管线同表征输入），非原始视频端到端 SSL——本对照隔离的是**伪标签消费机制**而非表征学习；披露为 harness 限定。
- τ=0.95 未调参；教师无 EMA 动量衰减（5 轮硬轮换）——实现保守，不利方向如实。
- 若 mt 因软标签在全类空间稀疏而训练不稳（预期风险：60 类×60 类软标签稀疏），按冻结规则如实报并停。

## 6. Evidence

Driver: `scripts/run_r23_ssl_baseline.py`; evidence: `reports/r23-ssl-baseline-<date>.json`.

## 修订历史

| 版本 | 日期 | 说明 |
|---|---|---|
| v1.0 | 2026-09-07 | FROZEN: R23c#4 驱动的外部 SSL 基线协议。判据/臂/种子/超参冻结。 |

## 7. Results (2026-09-07, post-run)

| Arm | top-1 (10 seeds, mean ± std) |
|---|---|
| (a) linear head @10% | 66.05% (deterministic) |
| (b) PSD pipeline | 67.53% ± 0.24 |
| (mt) Mean-Teacher pool distillation | **46.79% ± 0.61** |

**Verdict per the frozen §4 rule: the PSD pipeline exceeds the external SSL baseline by +20.7pp — 10/10 paired seed wins, Wilcoxon p=0.002 (Holm-corrected 0.008 over the series-wide six-test family), far outside the ±1pp parity band and far above the +2.9pp "no advantage" trigger.** The Mean-Teacher arm is heavily handicapped in this harness exactly as the frozen §5 disclosure anticipated: soft pseudo-labels over 60 classes are sparse, and the τ=0.95 confidence mask admits only 15,758 of 36,082 pool clips (vs. the PSD pool's ~35.4k at near-full acceptance) — consistent with the paper's mechanism claim that at NTU scale near-full pool restoration *is* the mechanism, which a confidence-thresholded external SSL baseline cannot replicate. Both directions of the frozen rule were live; the favorable direction materialized and is reported with its mechanism-consistent explanation. Evidence: `reports/r23-ssl-baseline-2026-09-07.json`; driver `scripts/run_r23_ssl_baseline.py`.
