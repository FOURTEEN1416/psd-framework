# P1.2 AK v2 复现/鲁棒层判读报告 — PSD-AKV2-PREREG-001 执行结果

> **指标口径：公开真实层 v2**（多段扩容，352 clips，train 256 / val 96，8 类有样本）| 日期: 2026-09-04
> 协议: 预注册冻结于构建前（`docs/paper/ak-v2-expansion-preregistration.md`）；v1 数字不替换，v2 为并列复现层
> 自动产物: `reports/p12-akv2-replication-2026-09-04.json`；构建 `reports` 侧 `runs/public_real_dataset/full12v2_*`

## 1. 数据集漏斗（EP1）

- 规划 357 合格段 → 提取成功 352（失败 5：段超出视频尾/全缺检）。
- 类别分布: watch 134 / track 83 / jump 47 / stay 34 / bite 24 / retrieve 16 / apprehend 12 / sit 2。
- **bark 零段过 0.80 一致性门**（犬吠帧与 watch/track 混标），sit 仅 2——v2 为 8 类空间，与 v1 9 类跨层比较仅方向性。
- 少数类扩容兑现: retrieve 8→16 / apprehend 7→12 / bite 8→24（v1→v2）。

## 2. 结果（n=10 seeds）

| 臂 | spc2 top-1 | spc2 macro-F1 | spc4 top-1 | full top-1 |
|---|---|---|---|---|
| PSD warm | **33.23±3.35** | 7.80±2.36 | **33.65±1.56** | **37.50** |
| AimCLR SSL | 20.31±1.72 | 1.71±0.43 | 20.52±2.20 | 18.75 |
| scratch | 23.44±1.79 | 1.32±1.48 | — | — |

配对 Wilcoxon（同 seed）：warm vs aimclr top-1 spc2 **+12.92pp 10-0-0 p=0.002** / spc4 +13.12pp 10-0-0 p=0.002；**macro-F1 双预算亦 10-0-0 p=0.002**（+6.08/+6.19pp）；warm vs scratch top-1 +9.79pp p=0.002。

## 3. 预注册判据执行结果

- **EP3 天花板检验: DATA_BOTTLENECK_CONFIRMED**——v2 full 37.50% − v1 33.93% = **+3.57pp ≥ +3.0pp 冻结阈值**。"绝对天花板由数据量主导、非方法缺陷"的归因从辩解升级为预注册检验成立的结论。
- **EP2 复现**: 管线核心结论在独立构建层复现且增强——低预算保留率 33.23/37.50 = **88.6%**（spc2，6% 标注比例）与 89.7%（spc4，12.5%）；warm-start 对通用 SSL 的优势从 v1 的 top-1 显著扩大到 **双指标显著**（AimCLR 在干净多段数据上坍缩: macro-F1 1.7%）。
- v1 的 macro-F1 持平判定为 v1 单片段结构伪影的候选解释: v2 逐帧一致性门剔除混标签段后，SSL 特征的少数类预测优势消失。

## 4. 诚实边界

1. v2 8 类空间与 v1 9 类不可直接合并报告；跨层数字仅作方向性对照。
2. v2 macro-F1 绝对值低于 v1（sit n=2 拖低逐类均值）——两指标口径随行披露。
3. Y_CKPT 见过 v1 训练视频，v2 向其训练池添加同视频新段（任务预训练固有，val 视频不相交，协议 §5 已披露）。
4. spc2 在 v2 为 6% 标注比例（绝对预算 2 片段/类与 v1 一致）——论文措辞用"matched absolute budget"，禁写"13% of labels"于 v2 行。

## 5. 论文回填位置

- §4.1 数据集段: v2 复现层一句话（协议 ID + 构建差异）。
- §4.3: 新增 E7b 复现段（EP2/EP3 结果 + 判据）。
- tab2: 新增 Public-real (v2 replication) 行。
- E7 段天花板句: 由"归因披露"升级为"预注册检验成立"。
- skeleton/outline: v1.5 登记。

## 6. 勘误附注（R8，2026-09-04）

- §3 "EP3 判据触发 DATA_BOTTLENECK_CONFIRMED" 修复：原始 +3.57pp 差值被类别空间混淆（12→8 类 chance 位移 +4.17pp 更大）。补同空间对照：v1 full 在 8 类子空间重算=25.96% → 同空间差 **+11.54pp**（above-chance 13.5→25.0pp）——数据瓶颈归因于 top-1 修复后成立；macro-F1 反向（14.7→7.8，sit n=2 驱动）双指标并报。协议 §7 追加 dated 修订。
- Y_CKPT 源视频泄漏披露按协议 §5 承诺补入正文（E7b 段）。


---

## Errata (2026-09-05, R16 adversarial review)

The end-to-end arms reported in this file trained the FINAL classifier on the true labels of the pseudo-labeled pool clips (and the iteration's precision-drop stopping consumed training-split labels). This is a protocol error: the reported budget percentages (13%/6%/10%) describe the seeds only, while the reported head consumed 31–100% (AK; one scratch arm pool covered the entire training split) / ≈96% (NTU) of training labels. The paper's end-to-end numbers are superseded by the corrected protocol: `reports/r16-endtoend-pseudo-2026-09-05.json` (AK v1/v2) and `reports/r16-ntu-pseudo-2026-09-05.json` (NTU). Pure-supervised arms (full-budget references, the EP3 ceiling test, the fine-tune controls) remain valid as archived. See `docs/paper/review-log.md` R16.


Additionally (R17): the archived macro-F1 values in this file were computed by an evaluator iterating `range(len(class_names))` over non-contiguous integer labels, dropping out-of-range classes (v2 full 7.83% is wrong; corrected value 19.33%, see `reports/r16-endtoend-pseudo-2026-09-05.json` warm_full). The "macro-F1 moves the other way" reading is superseded: under the corrected evaluator macro-F1 rises with top-1.
