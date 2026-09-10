# R32 指定视觉验收报告（投稿前 · designated visual-review）

- 日期：2026-09-11
- 角色：论文投稿前指定视觉验收审稿员（toolkit comp-visual-review 角色）
- 技能加载：`comp-visual-review/SKILL.md`（多模态视觉铁律 + --review 审核模式 + VERDICT 契约）、`scholar-critique-figures/SKILL.md`（per-figure 准则）——均全文读入并执行
- 验收对象：六图源文件（600dpi PNG + 矢量 PDF）+ main.pdf 五个载图物理页（5/11/16/21/32 ↔ 印刷页 p4/10/15/20/31）
- 全局裁决：**PASS**（fatal 0 / major 0 / minor 4）

---

## 1. 视觉工具实际调用证据（--review 审核模式，无降级）

运行方式：`cd 科研工具箱 && py -V:Astral/CPython3.11.15 tools/<tool> <fig>.png --review`
（注：记忆中的 `.venv311` 已不存在于工具箱根目录；Astral CPython 3.11.15 等价承载 pyc，属运行时替代非工具降级。视觉 API 走 .env 已配置 key，六次调用全部真实发出。）

| 图 | 工具 | 工具输出（原文摘录） | exit |
|----|------|---------------------|------|
| fig1_framework_overview | tikz_vision_check --review | `PASS` | 1（异常，见 V3） |
| fig2_pseudo_label_loop | tikz_vision_check --review | `ISSUE 1: [底部左侧区域] "AL queue" 和 "Verified seeds" 两个节点左侧及下方存在明显空白，上排与下排节点之间的垂直间距也较大，整体布局略显松散，不够紧凑。` | 1 |
| fig_ga_graphical_abstract | tikz_vision_check --review | `PASS` | 1（异常，见 V3） |
| fig3_segmentation_qualitative | data_fig_vision_check --review | `PASS` | 0 |
| fig4_al_efficiency | data_fig_vision_check --review | `PASS` | 0 |
| fig5_budget_retention | data_fig_vision_check --review | `PASS` | 0 |

- 未发生 API 不可用降级；tikz 工具 exit code 与文本裁决不一致（fig1/fig_ga 打印 PASS 却返回 1，data 工具正常 0）——以打印裁决文本为准，异常登记为 V3 工具备注。
- 工具通用提示词不覆盖 R30 海洋清风契约 A–D，故六图另由审稿员本人逐图多模态 Read 审查（见 §3），即"双通道"：工具证据 + 人工多模态复核，无凭感觉 pass。

## 2. Step 1 确定性预检（全过）

- PIL：六 PNG 全部解码成功，RGBA，600dpi（599.9988），尺寸 2043×1568 / 2043×1764 / 1956×1837 / 1940×2277 / 2371×1961 / 4449×1116。
- pdfimages -list：六个图 PDF **零嵌入位图**（纯矢量，v0.6 登记状态保持）；main.pdf 五个载图页同样零嵌入位图（矢量嵌入）。
- pdftotext 全文扫描：无 `??`、无 `TODO`。
- 图注映射（tex 真源核对，发现**文件名与论文图号交叉**，与任务书预告一致）：
  - `fig1_framework_overview.pdf` → Figure 1（01-introduction.tex:26）
  - `fig2_pseudo_label_loop.pdf` → Figure 2（03-method.tex:42）
  - `fig3_segmentation_qualitative.pdf` → Figure 3（04-experiments.tex:30）
  - `fig5_budget_retention.pdf` → **Figure 4**（04-experiments.tex:60，预算保留散点）
  - `fig4_al_efficiency.pdf` → **Figure 5**（05-ablation-analysis.tex:49，AL 两面板）
  - `fig_ga_graphical_abstract` 不在 main.pdf 内（独立图形摘要）✓

## 3. 逐图审查（scholar-critique per-figure 准则 + R30 契约 A–D）

### fig1_framework_overview（论文 Figure 1，p4）— PASS
| 检查 | 裁决 | 备注 |
|------|------|------|
| 格式/分辨率 | PASS | 矢量 PDF 主件 + 600dpi 预览 |
| 色板契约 A | PASS | 物理层浅青容器/语义层浅橙容器/浅绿接口带（embeddings + proposals），深青=物理、深橙=语义，同色同意图 |
| 契约 B | PASS | 无阴影/渐变/发光；深色满块恰 4 处（Unlabeled streams 入口、Rule-engine seeds 入口、Classification under 𝒴 出口、Taxonomy 𝒴→𝒴′ 注释带）=上限内 |
| 契约 D | PASS | 深底白字、白卡深字，印刷尺寸可读 |
| caption 对应 | PASS | Φ left/Ω right、embeddings+proposals 接口、𝒴→𝒴′ only Ω retrains 逐条与图一致 |
| 文字截断/重叠 | PASS | 无 |

### fig2_pseudo_label_loop（论文 Figure 2，p10）— PASS（1 minor）
| 检查 | 裁决 | 备注 |
|------|------|------|
| 格式/分辨率 | PASS | 矢量 + 600dpi |
| 色板契约 A | PASS | 橙描边=automated（Assign/Update Ω/Re-estimate P/Label pool 满块）、青描边=human-in-the-loop（AL queue/Verified seeds）、墨黑 hub（P/Ω/A 白字），与图注"Orange-labeled…teal-labeled…"逐字一致 |
| 契约 B | PASS | 深色满块 2 处 ≤4；虚线 write-back spokes 与图注一致 |
| caption 对应 | PASS | κ≥τ/κ<τ 路由、next round 顶回、verified seeds→hub 全对上 |
| 布局 | **minor（V1）** | 工具 ISSUE1 + 本人复核一致：下排左侧（AL queue/Verified seeds 周边及图例上方）留白偏大、上下排行距松。纯美学项，无遮挡/无截断/不改语义 |
| 文字可读性 | PASS | 印刷尺寸最小字（conf. κ / ≤B / each round）可读 |

### fig3_segmentation_qualitative（论文 Figure 3，p15）— PASS
| 检查 | 裁决 | 备注 |
|------|------|------|
| 格式/分辨率 | PASS | 矢量 + 600dpi；定性带为矢量绘制非位图截图 |
| 色板契约 A | PASS | 伪 GT 上带橙、SMQ 下带青、基线灰纹——与图注 orange/teal 承诺逐字一致 |
| 色盲安全 | PASS | 青橙对 + 灰纹基线带纹理冗余 |
| 轴/刻度 | PASS | frame index 0/1629/3258、0/1267/2535；Matched IoU 0–0.75 |
| 图例 | PASS | SMQ (E-C, K=8) / Random baseline 与系列一一对应，无框 |
| 数据一致性 | PASS | 图内 agg 0.4577±0.0488 · 4/4 > baseline 与图注/正文逐位一致；episode 标题 IoU 与柱值一致（0.420/0.540） |
| overplotting | PASS（备注） | SMQ 带密集边界是数据本身（预测段数量多），可辨读非缺陷 |

### fig4_al_efficiency（论文 Figure 5，p31）— PASS
| 检查 | 裁决 | 备注 |
|------|------|------|
| 格式/分辨率 | PASS | 矢量 + 600dpi |
| 面板标号 | PASS | (a) cold-start / (b) warm-started 与图注一致 |
| 色板契约 A | PASS | 橙=Uncertainty（automated 臂）、青=Random（契约深青=random 臂 ✓）、灰虚线=4.5% 基线；线型（实/虚）+marker（圆/方）双编码 |
| 轴 | PASS | 双 y 轴 0–100%，x 对数轴 20/50/100/200 与图注"log scale"一致 |
| 图注数字复核 | PASS | (a) b=100 青≈77.8>橙≈69.9、b=200 青≈88.0>橙≈80.9 目视可对；(b) 注释 4.2–5.0 pp 与曲线间距一致 |
| 碰撞/遮挡 | PASS | 图例、注释与数据无碰撞（R18 三处碰撞修复保持在位）；b=20 双 marker 重叠属数据重合（"within noise"如实呈现） |
| 负结果呈现 | PASS | 随机优于熵采样以注释+图注双写明，无美化 |

### fig5_budget_retention（论文 Figure 4，p20）— PASS
| 检查 | 裁决 | 备注 |
|------|------|------|
| 格式/分辨率 | PASS | 矢量 + 600dpi |
| 类别数 | WARN→可接受（V4） | 7 个图例项超 scholar-critique 5–6 指引；但仅 7 个数据点、双色系（青/橙家族）+7 种 marker 形状冗余，实际可辨识，不强制拆分 |
| 轴 | PASS | y=Retention of own full-budget top-1 (%)；x=log（% of tier's full-labeled pool）；10⁰/10¹/10² |
| 裁切 | PASS | v1 误差棒下帽（≈7%）完整可见（v0.6 y 下限修复在位）；PanAf 上帽 ≈104 在轴内 |
| 图注位移披露复核 | PASS | v1@13.9 / v2 spc4@11.5 / synthetic-offset@8.0 / UCF@8.8 / PanAf@13.2 / NTU120@16.5 与点位一致；NTU60 无可见误差棒=图注"±0.3pp smaller than marker"一致 |
| 图例 | PASS | 轴外下方两列、无框、与 marker/颜色一一对应 |
| 误导性轴 | PASS | log 轴位移已在图注逐点披露，保留率值不受影响 |

### fig_ga_graphical_abstract（独立图形摘要）— PASS
| 检查 | 裁决 | 备注 |
|------|------|------|
| 格式/分辨率 | PASS | 4449×1116 @600dpi 矢量 PDF |
| 色板契约 A/B | PASS | 青卡=Physics(frozen)、橙卡=Semantic(revisable, "taxonomy evolves 𝒴→𝒴′ only Ω retrains")、虚橙自环；无阴影/渐变；圆角家族一致 |
| 契约 D | PASS | 条形数值用墨黑（R24"中浅填充数值用墨色"纪律的落实，对 #53999D 对比 ≈6.4:1 优于白字 3.3:1）；canine 灰纹条+橙描边/橙数字 |
| 数字口径 | PASS | 90.7 / 88.9 / 28.9 与论文 Figure 4 及图注一致；脚注 canine 9.8 vs 11.1% chance 口径与正文 E7 一致 |

## 4. 载图页面审查（pdftoppm 150dpi，p5/11/16/21/32 物理页）— 5/5 PASS

| 页 | 印刷页 | 图 | 完整性 | caption 配对 | 碰撞 | 杂残留 |
|----|--------|----|--------|--------------|------|--------|
| 5 | p4 | Fig 1 | 不裁切 | 紧贴、编号-内容正确、与 tex 逐字一致 | 无（页眉页脚净空） | 无 ??/TODO/乱码，𝒴→𝒴′/Φ/Ω 渲染正常 |
| 11 | p10 | Fig 2 | 不裁切 | 同上 | 无 | 同上 |
| 16 | p15 | Fig 3 | 不裁切 | 同上（orange/teal 词与绘制一致） | 无 | 同上 |
| 21 | p20 | Fig 4 | 不裁切 | 同上（位移清单与点位一致） | 无 | 同上 |
| 32 | p31 | Fig 5 | 不裁切 | 同上（panel (a)/(b) 配对正确） | 无 | 同上 |

全局观察（V2，minor）：每页页脚为占位作者 "USER: surname et al.: Preprint submitted to Elsevier"——属已知投稿前人工项（作者信息），非图件缺陷，实际投稿前必须替换。

## 5. 跨图一致性（Step 4）— PASS

- 色板：六图全部落于海洋清风八色+tint+灰阶+墨黑域内；深青/深橙语义（物理 vs 语义、automated vs random、human-in-the-loop 青）跨图无冲突；灰=基线/null 一致（fig3 random baseline、fig4 4.5% 基线、GA canine 灰纹条）。
- 字体：六图统一 Arial 系无衬线；无栅格化文字（pdfimages 零位图佐证）。
- 数据图房子风格：fig3/fig4/fig5 top/right spine 全关、无框图例 ✓。
- 圆角/描边语言：fig1/fig2/GA 卡片圆角+家族色细描边一致；深色满块计数 fig1=4、fig2=2、GA=2（条形），均 ≤4。
- 图号-文件交叉命名（fig4↔Fig5、fig5↔Fig4）为仓库内部命名惯性，PDF 读者侧编号连续正确，无需改动。

## 6. Findings 分级清单

| ID | 严重性 | 位置 | 证据 | 修复建议 |
|----|--------|------|------|----------|
| V1 | minor | fig2_pseudo_label_loop 下排左侧+行间 | 视觉工具 ISSUE1 + 本人复核：AL queue/Verified seeds 周边及图例上方留白偏大、上下排行距松；无遮挡/截断 | 可选：压画布高度或上移图例收紧构图；不阻塞投稿 |
| V2 | minor | main.pdf 全部页脚 | "USER: surname et al." 占位作者名肉眼可见 | 投稿前替换真实作者/机构（已知人工项，非图缺陷） |
| V3 | minor | tikz_vision_check 工具 | fig1/fig_ga 打印 PASS 但 exit=1，与 data 工具（exit=0）不一致，疑工具退出码 bug | 工具箱侧排查退出码逻辑；本次以打印裁决文本为准 |
| V4 | minor（WARN） | 论文 Figure 4（fig5_budget_retention）图例 | 7 图例项超 scholar-critique 5–6 指引；双色系+七 marker 冗余缓解，实测可辨 | 可接受不改；若审稿人提可合并 minor tier 或移图例入轴旁 |

**fatal_count = 0；major = 0；minor = 4；status = PASS。**

## 7. 结论

六图 + 五个载图页全部通过出版级视觉验收：无文字不可读、无数据遮挡、无裁切、无 caption 与图矛盾（fatal 四项全零）；对比度、风格契约、图例对应均达标。视觉工具六次 --review 真实调用无降级，另以审稿员多模态逐图复核覆盖工具不检查的 R30 契约 A–D。允许进入下一步投稿流程；V1–V4 均为非阻塞项，其中 V2（作者占位）为投稿前必须完成的人工项。
