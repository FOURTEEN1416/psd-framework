# W22 任务书 — fig3/fig4 矢量图绘制（数据已齐，含负结果诚实呈现）

> 窗口: W22（全新独立窗口，纯绘图零 GPU）
> 日期: 2026-08-25 | 编制: 协调者歆歆
> 领地: `docs/paper/figures/**`（W17 收官移交；fig1/fig2 已完成勿动）
> 规范: `docs/paper/experiment-skeleton.md` 图表规范节（白底/#DADADA 网格/≤6 色/色盲安全蓝橙对/PDF 矢量/图内无标题）+ `figure-specs.md`

## 1. 任务目标（一句话）

绘制论文剩余两张数据图：fig3（SMQ 分割边界定性可视化）与 fig4（主动学习效率曲线——**负结果照画**），风格与 fig1/fig2 完全一致。

## 2. 必读输入

1. `dev-docs/HANDOVER.md` 快速启动节 + `AGENTS.md`
2. `docs/paper/figure-specs.md`（fig3/fig4 绘制规格）
3. `docs/paper/figures/FIGURE_SOURCE.md` + fig1/fig2 生成脚本（复用其 matplotlib 风格配置：字体/配色/线宽，保证全文视觉一致）
4. fig3 数据源: `reports/p02-2026-08-24.md` 及 E-C 相关 JSON（分割 episode 边界 vs 种子伪 GT）
5. fig4 数据源: `reports/p05-al-efficiency-short-2026-08-24.json`（curves_per_seed 字段：2 策略 × 4 预算 × 3 seeds）

## 3. 执行清单

### Step 1 — fig4 先行（纯数据图，确定性高）
- 双曲线：entropy vs random，x=标注预算 {20,50,100,200}，y=best_val_acc mean±std（误差棒 = 3 seeds std）
- 参考线：随机猜测基线 4.5%（22 类）
- **负结果如实呈现**：随机曲线在 b≥100 高于熵曲线的事实不隐藏不美化；caption 草稿写明"冷启动协议下不确定性采样未显示效率优势"
- 输出: `figures/fig4_al_efficiency.pdf` + `.png`(600dpi)
- caption 英文草稿附于 FIGURE_SOURCE.md

### Step 2 — fig3 定性可视化（需读 P0.2 产物结构）
- 内容：至少一个 episode 的 预测边界 vs 种子伪 GT 边界 对照（上下两条时间轴或同轴双色）
- 数据定位：从 `runs/p02_smq_eC/` 或 eval 产物中找边界序列；若可视化所需中间产物缺失，允许重跑推理脚本生成（CPU 可跑；GPU 排队纪律不适用——模型小）
- 若发现无可视化中间产物且重跑成本超 30min CPU：降级方案=改画边界 IoU per-episode 柱状分布（4 episodes 全超基线的事实本身就有展示价值），并在 FIGURE_SOURCE.md 登记降级理由
- 输出: `figures/fig3_segmentation_qualitative.pdf` + `.png`
- ⚠️ 替换现有 `fig3_placeholder.pdf`

### Step 3 — 一致性与溯源
- 两图过灰度打印自检（转灰度后曲线可区分）
- 更新 `FIGURE_SOURCE.md`：每图登记生成脚本路径、数据来源 reports 路径、绘制日期、caption 草稿
- 生成脚本落盘 `docs/paper/figures/scripts/`（可复现）

### Step 4 — 提交
- Conventional Commits 中文：`feat(paper): fig3 分割定性图+fig4 AL效率曲线(负结果如实)——矢量双格式+SOURCE登记`

## 4. 领地边界

**可写**: `docs/paper/figures/**`（除 fig1/fig2 本体）
**禁触**: `experiment-skeleton.md`/`outline.md`/`conclusion-limitations.md`（W21 领地，并行窗口）、一切代码/reports/dev-docs（只读）

## 5. 完成标准

- [ ] fig3_placeholder.pdf 被真图替换或降级方案留证
- [ ] fig4 含误差棒+随机基线线+负结果如实 caption
- [ ] 两图灰度自检通过、与 fig1/2 风格一致
- [ ] FIGURE_SOURCE.md 三要素齐全（脚本路径/数据路径/caption）

## 修订历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-25 | 建册：fig3/fig4 绘制（数据源与降级预案明确） |
