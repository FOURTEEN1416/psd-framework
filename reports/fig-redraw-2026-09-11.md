# fig-redraw-2026-09-11 — PSD 论文六图整体重绘（设计单 + 执行留痕）

> 任务：对 PSD 论文全部六张图按科研工具箱绘图技能链重绘（非修补）。
> 硬约束：psd_style.py（海洋清风八色板+语义映射）锁死；数据一个数不变（JSON 只读）；
> 印刷纪律（单栏 3.42in / GA 7.375×1.82in / Arial / 矢量 PDF+600dpi PNG / 深色满块 ≤4）；
> caption 仅最小同步；正文六节 tex 不碰；隔离纪律（技能库只读）。

## 〇、技能加载声明（开工首录）

| 角色 | 来源（数模竞赛·科研工具箱/） | 用途 |
|------|------------------------------|------|
| 总纲/选型 | CROSS_PROJECT_FIGURE_SKILLS_PROMPT.md | 五步法 + 隔离纪律 + 决策树 |
| 样式真源 | psd-framework docs/paper/figures/scripts/psd_style.py（R30 海洋清风） | 色板/语义/工艺唯一来源 |
| 概念图主参考 | skills/diagram-design（MIT） | fig1/fig2/fig_ga 构图、连线六规则、密度预算、remove test |
| 概念图借鉴（⚠只读，不执行其管线） | skills/paper-framework-figure-studio-pro | edge-label-first / no-false-relay / 模块化不碎片化 / 源忠实箭头原则 |
| 数据图总纲 | skills/paper-figure + references/pitfalls-and-intent.md（十八坑）+ composition-patterns.md（构图五模式） | fig3/fig4/fig5 选图三轴、P1–P18 逐条对照、图例/构图纪律 |
| 出版规范 | skills/scientific-visualization | 字号下限、色盲冗余编码、导出规格 |
| 收尾批判 | skills/scholar-critique-figures | 出图后逐图批判清单 |

**选型说明**：任务书给概念图两个候选 `paper-framework-figure-studio-pro`⚠ 或 `diagram-design`。
studio-pro 的 S0–S5 管线依赖 image_gen 生图路由、每阶段强制人工等待，且其自身规定"禁用
matplotlib/程序化渲染作为目标图产出"——与本任务"matplotlib 矢量管线 + psd_style 真源 + 自主执行"
根本冲突，故取 `diagram-design` 为主参考（任务书第二选项），studio-pro 仅按"⚠只读借用"取其
设计原则（变量上边、无假中继、源忠实），不执行其 S0–S5、不触发其门禁（隔离纪律第 3 条）。

**数据源核实（任务书表格勘误）**：
- fig3 任务表写 `p02-seg-strategy-ablation-2026-08-25.json`——经查该文件是分割**策略消融 arms**
  数据；fig3 时间轴+IoU 真源是现行脚本绑定的 `p02-smq-iou-eC-seeds-recheck.json`
  （aggregate 0.4577±0.0488 与 caption 逐位一致）。**维持现行绑定**。
- fig4/fig5 任务表两行数据源与脚本名交叉错位；以脚本实际绑定为准：
  fig4_al_efficiency ← `p05-al-efficiency-{short-2026-08-24, warmstart-short-2026-08-25}.json`；
  fig5_budget_retention ← `r16-endtoend-pseudo / r16-ntu-pseudo-10seed / p5b-ntu120 / p5b-ucf101 /
  p23-panaf / p07 / p12 / p14 / p05-warmstart-short`（决策口径 PanAf 取前 10 种子 42–51）。

## 一、重绘设计单（Step 1）

### fig1 框架总览 v11 → v12（概念图 · diagram-design）
- **新构图**：双层容器+浅绿接口桥拓扑不变。三处手术：
  1. 物理层两轨道（SSL→Dynamics / Quantization→Proposals）输出改"顶出→分高平移"正交路由
     （上行至 y=66/69 两条不同高度水平线，接入点间距 ≥12px），消除 v11 中 y=63 平线 ×
     x=33.5 竖线的连线交叉（diagram-design 连线规则 3 违例）；
  2. 接口桥竖排文字（diagram-design 反模式）废除 → 唯一接口箭头配**横排白底标签**
     "embeddings + proposals"（studio-pro edge-label-first：所载变量标在边上）；
  3. 深色满块 4→2：出口 Classification under 𝒴 + 底部演化带保留满块（焦点），
     入口 Unlabeled streams / Rule-engine seeds 降为白卡+家族色描边——恢复 psd_style
     "焦点强调每图 ≤2 处"纪律（v11 实际 4 处已超真源自定上限）。
- **改进点**：①消连线交叉 ②竖排文字→横排边标签 ③焦点数回归 ≤2、层级更清晰
  ④底部演化带与容器间距收紧，画布高度 2.62→约 2.55in，死区回收。
- **风险**：入口块视觉权重降低——以家族色描边+底部位置+箭头方向补偿；caption fig1 无颜色词、
  左右层位词不变，caption 零改动。

### fig2 伪标签循环 v14 → v15（概念图 · diagram-design Loop 型）
- **新构图**：双行拓扑不动（caption "top row/bottom row/along the top" 必须保真）。
  1. write-back 两条对角虚线（v14 对角连线，不可独立追踪）改正交圆角肘线：
     Update Ω 底边垂直下入 hub 顶；Re-estimate P 底边垂直下行→肘弯左行入 hub 右缘；
  2. "next round" 回路轨道从 y=82 压至贴盒上方（省 ~12% 高度反哺内容字号/盒高）；
  3. 图例重构：底部单行三键（automated / human-in-the-loop / write-back 虚线），
     删除 v14 中 "write-back" 在图内+图例的双处重复（保留 spokes 旁一处标注）；
  4. 六个站盒宽度统一 19.5（v14 Assign proposals 独宽 21.5 破坏节奏）。
- **改进点**：①对角线→正交肘（可追踪性）②顶部死区回收 ③图例去重+单行化 ④盒宽统一。
- **风险**：肘线圆角 rad 参数需目检；reest 肘线水平段与 verified→hub 实线分处 hub 两侧不交叉
  （几何已验算）；caption 色词 orange/teal 映射不变。

### fig3 分割时间轴 v6 → v7（数据图 · paper-figure 十八坑 P8/P10/P17）
- **新构图**：双 Episode 时间轴+四 episode 聚合柱面板拓扑不变。工艺升级：
  1. 每条轨道垫全程灰色底轨（GRAY_FILL、zorder 最低）——0..T 覆盖范围可见，空隙段从
     "无背景的漂浮"变为"可读的未覆盖区间"（补真信息）；
  2. 删面板 1 的 "frame index" 轴名（保留刻度），仅面板 2 保留——消重复轴名（反冗余）；
  3. 面板题精简 "Episode 1 · T=3258 · IoU 0.420"（删 "(baseline 0.291)"——数值已在柱面板
     与柱标签中，题目重复且过长）；
  4. 柱面板图例移入图区右上（frameon=False；上部空间富余），底部空间回收给刻度与轴名；
  5. 分段描边 0.6→0.5 降噪；画布高 3.45→约 3.3in。
- **改进点**：①底轨补覆盖信息 ②去重复轴名 ③面板题降噪 ④图例入图内、字号余量增大。
- **风险**：底轨灰与 baseline 灰柱同族——底轨无描边+在时间轴面板、灰柱有 hatch+在柱面板，
  语境分离；caption 的 orange/teal band 词与上下带顺序不变。

### fig4 AL 效率双面板 v4 → v5（数据图 · paper-figure P17 + composition 模式二）
- **新构图**：双面板 sharex 不变；图例升为**图级单行共享**（fig.legend，ncol=2，置于面板 (a)
  上方图外）——两面板系列完全相同，v4 双面板各挂一份图例纯属重复，且 (a) 图例压住 b=20
  数据点（P17 实锤违例）。
- **改进点**：①消 P17 图例遮点 ②去双面板图例冗余 ③注释/基线文字位置微调 ④高度 4.05→约 3.85in。
- **风险**：fig.legend 与 bbox_inches="tight" 的组合需逐像素复查防裁剪；caption 面板词不变。

### fig5 预算保留散点 v4 → v5（数据图 · paper-figure P4/P7 + critique 类目数）
- **新构图**：散点+误差棒+log x 轴不变；dodge 值、白描边、PanAf 前 10 种子口径、JSON 直读全部承 v4。
  1. **x 轴域收紧** (0.7,120)→(4,30)：v4 右半幅（20–120%）完全空置，主簇被压在中部 1/3；
     收紧后 5.5–16.5% 的点簇横向展开近一倍，可辨性大增；刻度改 [5,10,20]；
  2. **图例压缩**：ncol=2×4 行长标签 → ncol=4×2 行短标签（"human bench. NTU60" 等），
     底部占高降 ~40%，释放高度反哺绘图区；
  3. ylim (2,112)→(0,108)：保留率从 0 起步（P4 诚实轴），顶部死区回收。
- **改进点**：①消右半空版 ②图例占高减半 ③0 起步轴。
- **风险**：x 域收紧后 13.9(v1)/13.2(PanAf) 水平间距按 log 距离**变大**（分母变小），whisker
  净空只增不减；caption 的位移清单数值不动。

### fig_ga 图形摘要 v11 → v12（概念图 · diagram-design remove test）
- **新构图**：三段式（Φ 块 → 箭头 → Ω 块 → 保留率微条）与规格 7.375×1.82in 不变。
  1. 删 Ω 块内装饰性弧线自环（remove test：不承载信息；"taxonomy evolves 𝒴→𝒴′: only Ω
     retrains" 文字行已完整表达）；
  2. Ω 块补机制行 "anchor-guided pseudo-labeling · self-training"（与 Φ 块密度平衡，
     内容源忠实于 fig1/方法节）；
  3. 双块文字垂直重排消中空死区；脚注 4.6→5.1pt 并精简为单行+短注。
- **改进点**：①删装饰冗余 ②密度平衡至 ~4/10 ③脚注达到可读下限。
- **风险**：GA 不进 main.tex（独立投稿资产），无 caption 耦合；保留率三条数值 90.7/88.9/28.9 不动。

## 二、执行与验收留痕（Step 2–4）

### 2.1 产出清单（版本号递增，产物覆盖 docs/paper/figures/ 同名文件）

| 图 | 新脚本 | 新版本要点 |
|----|--------|-----------|
| fig1 | make_fig1_overview_v12.py | 零交叉正交路由 + 横排接口边标签 + 焦点深块 4→2 |
| fig2 | make_fig2_pseudo_label_loop_v15.py | write-back 正交肘线 + 回路轨道贴盒(高 2.95→2.35in) + 单行三键图例 + 盒宽统一 |
| fig3 | make_fig3_segmentation_qualitative_v7.py | 全程灰底轨 + 去重复轴名 + 面板题精简 + 图例入图内(5.5→5.8pt) |
| fig4 | make_fig4_al_efficiency_v5.py | 图级单行共享图例(消 P17 压点) + 高度 4.05→3.66in(消浮动超页) |
| fig5 | make_fig5_budget_retention_v5.py | x 域 (0.7,120)→(4,30) 刻度[5,10,20] + 图例 4 列×2 行短标签 + ylim 0 起步 |
| fig_ga | make_ga_graphical_abstract_v12.py | 删装饰自环 + Ω 块补机制行密度平衡 + 脚注 4.6→5.1pt + **pad 归零使 PDF 精确 531×131.04pt** |

### 2.2 就地迭代记录（批判/验收驱动，3 处）

1. **fig5 刻度 bug**：自定义 xticks 后 log 轴 minor formatter 在边界(4/6/30)打出
   "4 × 10⁰" 混排标签 → `minorticks_off()` + `NullFormatter()` 根治；
2. **GA 尺寸超限**：`bbox_inches="tight"+pad 0.02` 使 PDF 实际 533.88×133.92pt，
   超 Elsevier 531×131 上限 2.9pt（v11 即如此，印的"531×131"是名义值）→ pad 归零，
   实测 **531 × 131.04 pt** 达标；
3. **fig4 浮动超页**：首版编译 `Float too large for page by 8.25pt`（280.9pt 高）
   → figsize 3.85→3.66in + pad 0.08→0.04（字号 pt 固定不受影响）→ 警告清零。

### 2.3 scholar-critique-figures 批判清单结果（逐图）

程序化预检：六 PDF 经 `pdfimages -list` 全部**零嵌入位图**（纯矢量）✓；PNG 600dpi ✓；
无红绿对、无 rainbow/jet、无双 Y 轴、无饼图/3D、误差类型均在 caption 申明（seed std）✓。

| 图 | 格式/分辨率 | 色盲风险 | 类目数 | 字号(印刷有效) | 过散点 | 动力柱 | 判定 |
|----|------------|---------|--------|---------------|--------|--------|------|
| fig1 | PASS(矢量) | PASS(青/橙+位置冗余) | PASS(2 语义族) | PASS(≥5.5pt) | N/A | N/A | PASS |
| fig2 | PASS | PASS(橙/青绿+描边+图例键) | PASS(2 族+hub 墨) | PASS | N/A | N/A | PASS |
| fig3 | PASS | PASS(橙/青+上下带位+粗彩轴标) | PASS(2 系列) | PASS(刻度5.5pt×1.05) | N/A | PASS*(n=4 全量展示,无均值遮蔽) | PASS |
| fig4 | PASS | PASS(实心圆/空心方+实/虚线) | PASS(2 系列) | PASS | N/A | N/A | PASS |
| fig5 | PASS | PASS(7 形状完全冗余编码) | **WARN(7 类>5-6)** | PASS(图例5.3pt×1.10≈5.8pt) | PASS(7 点) | N/A | PASS(WARN 已论证) |
| fig_ga | PASS | PASS | PASS(3 条) | PASS(脚注5.1pt) | N/A | N/A | PASS |

**fig5 类目数 WARN 论证**：7 层级皆为 caption 逐条主张的实验结论，不可合并；缓解=
每系列 marker 形状唯一（o/s/D/v/P/X/^ 七形完整冗余，形状单独即可辨识）+ 色板语义由
psd_style 真源锁死（R26→R30 用户裁决链）。合并类目将违反硬约束 1，故记录 WARN 不强改。

### 2.4 Step 3 编译与断言（pdflatex 绝对路径 ×2）

- 编译：pass1/pass2 exit=0；`main.log` 错误 **0**；浮动警告 **0**；**42 页**；
- `??` 未解析引用：**0**；
- 图 1–5 印刷页落位（印刷页=物理页−1）：**p4 / p10 / p15 / p20 / p31 全部命中，零回退**
  （Fig5 caption 所在物理页 32 页脚直证 "Page 31 of 42"）；
- caption 颜色词 ↔ 图一致性：fig2 "Orange-labeled … teal-labeled" ↔ 自动步深橙描边/
  人工步青绿描边 ✓；fig3 "orange band / teal band" ↔ 上橙下青 ✓；
- 全文 USER 残留：**0**（pdftotext 全文扫描）；
- caption/正文六节 tex：`git diff docs/paper/latex/` 为空 = **零改动**。

### 2.5 Step 4 逐图 150dpi 实看（页 5/11/16/21/32 + 600dpi 原生 zoom 双核）

| 页(印刷) | 图 | 文字可读 | 无遮挡 | 无截断 | 图例对应 | caption 配对 |
|---------|----|---------|--------|--------|---------|-------------|
| p4 | fig1 | ✓(400dpi zoom: 层标题/流线净空足) | ✓ | ✓ | ✓(接口边标签) | ✓ 左右层/只经 embeddings+proposals |
| p10 | fig2 | ✓(400dpi zoom: κ≥τ 与两侧盒缘净空清晰) | ✓ | ✓ | ✓(单行三键) | ✓ top/bottom row 措辞保真 |
| p15 | fig3 | ✓ | ✓ | ✓ | ✓(图内右上) | ✓ orange/teal 带与聚合面板 |
| p20 | 论文图4(fig5) | ✓ | ✓ | ✓ | ✓(4列×2行) | ✓ 位移清单/NTU60 bar<marker |
| p31 | 论文图5(fig4) | ✓ | ✓(图例已移出面板) | ✓ | ✓(图级单行) | ✓ (a)/(b) 负结果注记 |
| 独立 | fig_ga | ✓(600dpi) | ✓ | ✓(531×131.04pt 达标) | ✓ | 不进 main.tex，独立资产 |

已知伪影（非缺陷）：pdftotext 把 caption "efficiency" 的 ffi 连字抽成 "eciency"
（cas-sc 字体连字→提取丢失），页面渲染 150dpi 实看字形完整，属提取器伪影。

## 三、caption 最小同步清单

**零改动**。全部构图重设计均以"caption 措辞保真"为前置约束：
fig1 caption 无颜色词、left/right 层位不变；fig2 保留 top/bottom row 拓扑词；
fig3 orange/teal 带与上下顺序不变（baseline 数值从面板题移入柱面板，信息不灭）；
fig4/fig5 面板词、位移清单、误差申明全部承旧；fig_ga 不进 main.tex。
`git diff docs/paper/latex/` 为空即证明。

## 四、逐图前后对比说明

### fig1 v11 → v12
v11 病灶：物理层两轨道输出线在 (33.5, 63) 处横竖交叉（diagram-design 连线规则 3 违例）；
接口桥竖排文字（反模式清单在案）；深色满块 4 处超 psd_style 自定"焦点 ≤2"上限。
v12：输出改"顶出→左上右下分高平移"（y=68.8/65.4），左横线越右竖线顶端，全图零交叉；
接口箭头配横排两行白底标签（edge-label-first）；入口两块降白卡+家族色描边，焦点=
出口 Classification + 演化带 = 2。信息量不变，层级信号更强。

### fig2 v14 → v15
v14 病灶：write-back 两条对角虚线不可独立追踪；回路轨道 y=82 吊高吃掉 ~20% 高度；
write-back 图内+图例双处重复；Assign 盒独宽 21.5 破节奏。v15：正交肘线（upd 垂直下入
hub 顶；reest 垂下→左肘入 hub 右缘，与 verified→hub 实线分处 hub 两对边不交叉）；
轨道贴盒 y=69.5；图例单行三键；盒宽统一 19.5；画布 2.95→2.35in（顶部死区全回收）。
拓扑未动，caption 的 top/bottom row 措辞逐词保真。

### fig3 v6 → v7
v6 病灶：无底轨→分段漂浮、空隙段信息不可读；"frame index" 轴名两面板重复；面板题带
"(baseline …)" 与柱面板重复；图例吊在轴下方占高度。v7：全程灰底轨补覆盖信息；面板 1
删轴名；题精简；图例入图内右上且字号 5.5→5.8pt；画布 3.45→3.30in。

### fig4 v4 → v5
v4 病灶（十八坑 P17 实锤）：面板 (a) 图例压住 b=20 数据点；两面板系列相同却各挂一份
图例。v5：图例升图级单行共享（面板上方图外）；面板内仅留负结果注记与基线标签；
高度按浮动超页警告收敛 4.05→3.66in。负结果叙事（random ≥ uncertainty）原样保留。

### fig5 v4 → v5
v4 病灶：xlim (0.7,120) 下右半幅全空、主簇挤在中部 1/3；图例 2 列×4 行长标签占底部
~40% 高度；ylim 从 2 起步。v5：x 域 (4,30)+刻度 [5,10,20]，点簇横向展开近一倍且
13.9/13.2 的 log 间距变大（whisker 净空只增）；图例 4 列×2 行短标签；ylim (0,108)
诚实轴。dodge 数值/白描边/PanAf 前 10 种子口径/JSON 直读逐位承 v4。

### fig_ga v11 → v12
v11 病灶：Ω 块装饰性弧线自环不承载信息（remove test 不过）；双块中空死区；脚注 4.6pt
低于可读下限；PDF 实际尺寸超上限 2.9pt。v12：删自环；Ω 块补机制行
"anchor-guided pseudo-labeling · self-training"（源忠实）平衡密度至 ~4/10；
文字重排；脚注 5.1pt；pad 归零后实测 **531×131.04pt** 精确达标。

## 五、技能来源署名

- 总纲与隔离纪律：数模竞赛·科研工具箱/CROSS_PROJECT_FIGURE_SKILLS_PROMPT.md（五步法）
- 概念图（fig1/fig2/fig_ga）：数模竞赛·科研工具箱/**diagram-design**
  （连线六规则/focal rule/密度预算/remove test/反模式清单）
- 概念图原则借鉴（⚠只读未执行管线）：数模竞赛·科研工具箱/**paper-framework-figure-studio-pro**
  （edge-label-first/no-false-relay/模块化不碎片化）
- 数据图（fig3/fig4/fig5）：数模竞赛·科研工具箱/**paper-figure**
  （选图三轴/十八坑 P1–P18/构图五模式 composition-patterns）
- 出版规范：数模竞赛·科研工具箱/**scientific-visualization**（字号下限/冗余编码/导出）
- 收尾批判：数模竞赛·科研工具箱/**scholar-critique-figures**（§2.3 逐图清单）
- 样式真源：docs/paper/figures/scripts/psd_style.py（海洋清风八色板，色值/语义零改动）

隔离纪律遵守声明：技能库全程只读（仅 Read），零写入；STEP_MANIFEST/门禁未触发未伪造；
一切产物落本仓。

## 六、任务书勘误留痕

1. fig3 数据源：任务表写 `p02-seg-strategy-ablation-2026-08-25.json`（实为策略消融
   arms 数据）；本图真源 = `p02-smq-iou-eC-seeds-recheck.json`（聚合 0.4577±0.0488
   与 caption 逐位一致），维持现行绑定。
2. fig4/fig5 任务表两行数据源与脚本名交叉错位；按脚本实际绑定执行。
3. psd_style.py 实际路径为 `docs/paper/figures/scripts/psd_style.py`（任务书写的
   `docs/paper/latex/figures/scripts/` 不存在）。
4. v11 的 GA 名义尺寸 531×131pt 实为 533.9×133.9pt（pad 撑大）；v12 严格化到
   531×131.04pt。
