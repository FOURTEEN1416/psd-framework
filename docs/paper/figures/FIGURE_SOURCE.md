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


## 2026-09-04 重绘登记（期刊规范优化, scientific-visualization 规程）

| 图 | 版本 | 重绘脚本 | 主要修复 |
|----|------|---------|---------|
| fig1 | v4 | `figures/scripts/make_fig1_overview_v2.py` | 双向箭头改单向数据流/embeddings+proposals 走接口带顶部通道（不再斜穿边界）/Y→Y' 乱码修复（raw string）/演化带文字与虚线分离/分类框贴层/聚类自迭代环/去 FIGURE_SOURCE 水印（移至本登记） |
| fig2 | v2 | `figures/scripts/make_fig2_pseudo_label_loop_v2.py` | κ<τ 分支直达 AL queue（原飞线跳过 queue）/判断节点入环/verified seeds 回流改指 re-estimate（原错指 init）/配色与 fig1 统一（青=人工种子侧, 橙=自动学习侧）/frozen 句移除（caption 已述） |

三表（tab1/tab2/tab3）同轮修复: tabular -> tabularx 版心自适应（原 tab1/tab2 右列被页缘裁切、tab3 碎行 12 段）。

| v0.3 | 2026-09-05 | **R9-R11 印刷尺寸重设计轮（judge 视觉门 4/4 通过）**：R8 视觉验收发现四图按大画布设计、缩至 8.6cm 栏宽后落地字号 2.0-2.6pt 不可读——全部改为**最终印刷尺寸设计**（figsize≈3.4in 宽，字号 1:1 落地 5.5-7.6pt）。现行脚本：fig1=make_fig1_overview_v3.py、fig2=make_fig2_pseudo_label_loop_v5.py（人工侧中性灰消除与 fig1 青=物理层跨图语义冲突；辐条标签贴 hub 外沿）、fig3=make_fig3_segmentation_qualitative_v2.py（去 zoom 列，caption 删 band-text 承诺——GT 段宽 21-148 帧物理放不下类名）、fig4=make_fig4_al_efficiency.py（去顶/右 spine，基线注释移虚线下方）。数据溯源不变（p02 recheck / AL JSON，聚合值逐位一致）。judge 三轮复验记录见 review-log.md R9-R11。 |

| v0.4 | 2026-09-05 | **R13b/R14 内容级修复轮（judge 三轮，5/5 视觉门通过）**：fig1 补 warm-start 标记+接口箭头改指 clustering 盒+自迭代弧右移防钩状交叠；fig2 辐条语义修正（仅 Ω/P/A 三真写回站，κ≥τ/κ<τ 路由标签移环弧外侧，focal 移至 Label pool 站，hub 加高防裁切）；fig3 仅 caption 修正（hardest→lowest/highest-IoU、agreement κ→confidence、补 label-agnostic 句）；**新增 fig5_budget_retention.py（渲染为 Figure 4）**：预算-保留率跨层散点，5 点 4 层直读 p10/p12/p14/w23/Y JSON 零硬编码，图例轴外下方；fig4_al_efficiency 增面板 (b) warm-start 臂（更强负证据）+基线注释白底+en-dash。三图正文引用补齐。坑：副题删除正则误吞 PDF savefig 行——复验必须核对工件字节。 |

| v0.5 | 2026-09-05 | **R16 协议修正重绘（仅 fig5）**：R16 对抗审稿实锤端到端臂协议错误（最终头消费池片段真标签+oracle 停止），AK v1/v2 与 NTU 点改读修正协议工件 `r16-endtoend-pseudo-2026-09-05.json` / `r16-ntu-pseudo-2026-09-05.json`（全预算分母仍读纯监督归档 p07/p12/p14——不受协议错误影响）；y 轴 78-104→15-112（AK 保留率坍缩至 28.9/35.0/44.2%，NTU 90.6±0.2，synthetic-offset 84.9±4.5 不变）；caption 从"near-flat 低资源主张"改为诚实的 tier-dependent 叙事；修复 `syn_full` 硬编码回退（改 `y_full["summary"]["best_val_acc"]` 真读）。工件核验：mtime + pymupdf 抽取 y 轴刻度 20..100 确认更新。fig1-fig4 本轮零改动。 |
