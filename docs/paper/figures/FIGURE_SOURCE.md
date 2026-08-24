# FIGURE_SOURCE — 论文图表溯源清单（W17）

> Owner: `docs/paper/figures/` · W17 窗口 2026-08-24
> 执行标准: `experiment-skeleton.md` §图表规范（白底浅灰 / ≤6 色 / 低饱和淡彩 / 避红绿 / 矢量输出）
> 绘制规格唯一来源: `docs/paper/figure-specs.md`（改图先改规格）

| 图 | 产物 | 生成脚本 | 规格来源 | 数据依赖 | 状态 |
|----|------|---------|---------|---------|------|
| fig1 框架总览（hero） | `fig1_framework_overview.pdf`（矢量）+ `.png` 预览 | `scripts/make_fig1_overview.py` | figure-specs.md §fig1 | 无（纯架构示意） | ✅ 已产出 |
| fig2 语义迭代闭环 | `fig2_pseudo_label_loop.pdf`（矢量）+ `.png` 预览 | `scripts/make_fig2_pseudo_label_loop.py` | figure-specs.md §fig2 | 无（算法流程示意） | ✅ 已产出 |
| fig3 SMQ 分割 vs GT | `fig3_placeholder.pdf` | `scripts/make_fig34_placeholders.py` | figure-specs.md §fig3 | ⏳ P0.2 实验数据 | 占位 |
| fig4 主动学习效率曲线 | `fig4_placeholder.pdf` | `scripts/make_fig34_placeholders.py` | figure-specs.md §fig4 | ⏳ P0.5 实验数据 | 占位 |

## 复现命令（仓库根目录执行）

```powershell
python scripts/make_fig1_overview.py
python scripts/make_fig2_pseudo_label_loop.py
python scripts/make_fig34_placeholders.py
```

## 绘图纪律

- 每张 PDF 右下角内嵌 `FIGURE_SOURCE:` 小注，指向本清单与生成脚本。
- 配色：物理层/种子链 = 淡青 `#DAFFFF` 系；语义层/处理节点 = 淡橙 `#FFE3DA` 系；接口带 = `#DADADA`；文字黑。两族深色描边（`#0E7490` / `#C2410C`）仅用于强调路径（Y→Y′ 虚线、κ 分叉），不引入新色相。
- fig3/fig4 正式图：待 P0.2/P0.5 数据落档后，在 figure-specs.md 补齐规格（fig3 行数与 GT 标注方式、fig4 折数与误差棒来源），再由后续窗口重绘替换占位 PDF。

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-24 | W17：fig1/fig2 矢量终图 + fig3/fig4 占位 + 脚本三件套入库 |
