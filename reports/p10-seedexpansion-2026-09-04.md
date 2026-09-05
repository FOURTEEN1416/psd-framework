# P1.0 种子扩容判读报告 — E7/P0.8 关键对比 n=3 → n=10

> **指标口径：公开真实层**（AK full12）| 日期: 2026-09-04 | 自动产物: `reports/p10-seedexpansion-2026-09-04.json`
> 协议: p07/p08 端到端管线逐字固定，仅扩种子集 42..51；scratch 臂每 seed 独立随机骨干（torch.manual_seed 可复现）

## 1. 结果（n=10）

| 臂 | spc2 top-1 | spc2 macro-F1 | spc4 top-1 | spc4 macro-F1 |
|---|---|---|---|---|
| PSD warm | **31.96±2.45** | 14.36±2.86 | **31.43±4.23** | 14.72±3.58 |
| AimCLR SSL | 26.07±4.55 | 14.78±4.00 | 24.11±5.60 | 10.87±3.59 |
| scratch | 24.64±1.64 | 4.39±0.17 | — | — |

## 2. 配对检验（同 seed 差，Wilcoxon signed-rank）

| 对比 | 指标 | Δ | 胜负 | p | 判定 |
|---|---|---|---|---|---|
| warm vs aimclr | top-1 spc2 | +5.89pp | **10-0-0** | **0.002** | 显著 |
| warm vs aimclr | top-1 spc4 | +7.32pp | 7-0-3 | **0.016** | 显著 |
| warm vs aimclr | macro-F1 spc2 | −0.42pp | 5-5-0 | 0.770 | 持平 |
| warm vs aimclr | macro-F1 spc4 | +3.84pp | 8-2-0 | 0.084 | 方向偏 warm 未显著 |
| warm vs scratch | top-1 spc2 | +7.32pp | **10-0-0** | **0.002** | 显著 |
| warm vs scratch | macro-F1 spc2 | +9.97pp | **10-0-0** | **0.002** | 显著 |

## 3. 判读与论文影响

1. **P0.8 消融从"方向一致未达显著"升级为"全预算 top-1 显著"**——审稿人批 n=3 的最易缴械条款解除。
2. **macro-F1 反转消解**: n=3 时 AimCLR spc2 反超（15.21 vs 12.44）在 n=10 下收敛为统计持平（p=0.77）——原"多数类锐化权衡"叙事降级为"种子噪声"，论文措辞已同步（tab3 行 + §4.4 段）。
3. **E7 主数字刷新**: warm@spc2 = 31.96±2.45（n=10），占全监督 33.93% 的 **94%**（原 91%）；scratch = 24.64±1.64；Abstract/Intro/§4.3/tab2 联动更新。
4. 措辞纪律: p=0.002/0.016 可写 significant；macro-F1 spc4 p=0.084 仍禁写显著。


---

## Errata (2026-09-05, R16 adversarial review)

The end-to-end arms reported in this file trained the FINAL classifier on the true labels of the pseudo-labeled pool clips (and the iteration's precision-drop stopping consumed training-split labels). This is a protocol error: the reported budget percentages (13%/6%/10%) describe the seeds only, while the reported head consumed 60–99% (AK) / ≈96% (NTU) of training labels. The paper's end-to-end numbers are superseded by the corrected protocol: `reports/r16-endtoend-pseudo-2026-09-05.json` (AK v1/v2) and `reports/r16-ntu-pseudo-2026-09-05.json` (NTU). Pure-supervised arms (full-budget references, the EP3 ceiling test, the fine-tune controls) remain valid as archived. See `docs/paper/review-log.md` R16.
