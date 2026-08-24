# FIGURE_SOURCE — 论文图表溯源清单（W17 建册 / W22 更新）

> Owner: `docs/paper/figures/` · W17 窗口 2026-08-24 建册 · W22 窗口 2026-08-25 补齐 fig3/fig4
> 执行标准: `experiment-skeleton.md` §图表规范（白底浅灰 / ≤6 色 / 低饱和淡彩 / 避红绿 / 矢量输出）
> 绘制规格唯一来源: `docs/paper/figure-specs.md`（改图先改规格；W22 规格偏差见下方登记节）

| 图 | 产物 | 生成脚本 | 规格来源 | 数据依赖 | 状态 |
|----|------|---------|---------|---------|------|
| fig1 框架总览（hero） | `fig1_framework_overview.pdf`（矢量）+ `.png` 预览 | `scripts/make_fig1_overview.py`（仓库根 scripts/） | figure-specs.md §fig1 | 无（纯架构示意） | ✅ 已产出 |
| fig2 语义迭代闭环 | `fig2_pseudo_label_loop.pdf`（矢量）+ `.png` 预览 | `scripts/make_fig2_pseudo_label_loop.py`（仓库根 scripts/） | figure-specs.md §fig2 | 无（算法流程示意） | ✅ 已产出 |
| fig3 SMQ 分割 vs GT | `fig3_segmentation_qualitative.pdf`（矢量）+ `.png`(600dpi) | `docs/paper/figures/scripts/make_fig3_segmentation_qualitative.py` | figure-specs.md §fig3 + W22 登记节 | `reports/p02-smq-iou-eC-seeds-recheck.json`（E-C 定稿，含逐 episode gt/pred 边界序列） | ✅ 已产出（替换占位） |
| fig4 主动学习效率曲线 | `fig4_al_efficiency.pdf`（矢量）+ `.png`(600dpi) | `docs/paper/figures/scripts/make_fig4_al_efficiency.py` | figure-specs.md §fig4 + W22 登记节 | `reports/p05-al-efficiency-short-2026-08-24.json`（W14 归档，curves 字段） | ✅ 已产出（替换占位） |

## 复现命令（仓库根目录执行）

```powershell
# fig1/fig2（W17 产物，脚本位于仓库根 scripts/）
python scripts/make_fig1_overview.py
python scripts/make_fig2_pseudo_label_loop.py
# fig3/fig4（W22 产物，脚本随图入库 docs/paper/figures/scripts/）
python docs/paper/figures/scripts/make_fig3_segmentation_qualitative.py
python docs/paper/figures/scripts/make_fig4_al_efficiency.py
```

> 注：`scripts/make_fig34_placeholders.py`（W17 占位脚本，仓库根）保留不改（非本窗口领地），
> 其占位产物 `fig3_placeholder.pdf` / `fig4_placeholder.pdf` 已于 W22 删除并由真图替换。

## W22 规格偏差与决策登记（2026-08-25）

1. **fig4 不画 figure-specs §fig4 待定项中的 "目标线 y=85% 虚线"**：该验收线属真实 K9 层口径；
   本数据为合成层短预算协议（JSON `meta.layer_note` 明确禁止由本实验外推 ≥85% 结论，硬规则"三层口径"）。
   按任务书画 4.5% 随机猜测基线（22 类）。
2. **fig3 数据源用 `p02-smq-iou-eC-seeds-recheck.json`**（argmax 平局 bug 修复后的复核版）；
   其聚合值与原版 `p02-smq-iou-eC-seeds.json` 完全一致（0.4577±0.0488），P0.2 报告 §8.3 定稿数字。
3. **fig3 未触发降级预案**：recheck JSON 内已含逐 episode `gt_segments`/`pred_segments` 边界序列，
   直接读 JSON 绘制，无需重跑推理。主展示 ep1（受分辨率压制最难）+ ep4（最优），避免只挑最好 episode；
   底部 IoU 柱状面板覆盖全部 4 episodes（4/4 超随机基线），防樱桃采摘质疑。
4. **fig4 负结果如实呈现**：随机曲线在 b≥100 高于熵曲线（77.8% vs 69.9% @b=100；88.0% vs 80.9% @b=200），
   图内以浅灰注释显式指出，caption 写明"冷启动协议下不确定性采样未显示效率优势"。
5. fig1/fig2 的生成脚本在仓库根 `scripts/`（W17 先例）；W22 起新图脚本按任务书落盘
   `docs/paper/figures/scripts/`（图与脚本同域，便于溯源）。

## Caption 草稿（自足式，英文）

> **Figure 3: Qualitative segmentation comparison on InterPet4D (public-real layer).** For two evaluation
> episodes, the upper track shows seed pseudo-ground-truth segments from the rule engine (κ ≥ 0.8, duration
> ≥ 0.5 s; band text = behavior class) and the lower track shows motion-word boundaries from SMQ (end-to-end
> K=8 codebook, epoch-30 checkpoint). Dashed boxes mark the zoom-in windows, where locally aligned boundaries
> are visible. The bottom panel reports per-episode matched IoU under the seed-pseudo-GT protocol for all four
> episodes: SMQ exceeds the random-segmentation baseline on 4/4 episodes (0.458 ± 0.049 vs. ≈ 0.30). Seed
> pseudo-GT is a weak, rule-derived reference rather than human annotation.

> **Figure 4: Active-learning efficiency on the synthetic layer (short-budget protocol).** Best validation
> accuracy (22-class top-1 on a fixed ground-truth validation set, n = 330; mean ± std over 3 seeds) versus
> annotation budget, with cold-start retraining at every budget point. Both arms saturate far above the 4.5%
> random-guess baseline (22 classes). Under this cold-start protocol, uncertainty sampling (softmax entropy)
> shows no efficiency advantage over random selection: it does not outperform random at any budget and is
> exceeded by random for budgets ≥ 100 (e.g., 69.9% vs. 77.8% at b = 100; 80.9% vs. 88.0% at b = 200).

## 绘图纪律

- 每张 PDF 右下角内嵌 `FIGURE_SOURCE:` 小注，指向本清单与生成脚本。
- 配色：物理层/预测段 = 淡青 `#DAFFFF` 系（描边 `#0E7490`）；语义层/种子 GT = 淡橙 `#FFE3DA` 系（描边
  `#C2410C`）；接口带/基线 = `#DADADA`；文字黑。≤6 色，色盲安全青橙对，灰度自检双编码（线型/标记/纹理）。
- 数据图一律从 `reports/` 归档 JSON 读数绘制，脚本内不硬编码实验数字；每次重跑脚本打印当次读数留证。

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-24 | W17：fig1/fig2 矢量终图 + fig3/fig4 占位 + 脚本三件套入库 |
| v0.2 | 2026-08-25 | W22：fig3/fig4 真图替换占位（脚本随图入库）；caption 草稿；规格偏差与决策登记 5 条；灰度自检通过 |
