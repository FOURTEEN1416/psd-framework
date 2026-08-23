# PSD-Framework 论文全文大纲（P0.6 初稿骨架）

> **Owner**: `docs/paper/`（唯一写作领地）· 创建: W5 窗口 2026-08-23
> **目标期刊**: Pattern Recognition（IF≈8）/ 备选 IJCV
> **方法论参照系**: 数模竞赛学术工具集（`galaxy-ml-paper-writing` 主叙事原则 + `paper-plan` 大纲规范 + `参考论文使用指南` 统计规律）。适配性裁剪记录见 §9。
> **状态标注约定**: ✅ 材料已齐备可写 ｜ ⏳ 待 P0.x 实验数据 ｜ 🚫 明确不做（本阶段范围外）

---

## 0. 一句话贡献（One-Sentence Contribution）

> We propose a physics–semantics decoupled framework for low-resource animal behavior recognition, which couples label-free skeleton-dynamics modeling (self-supervised pretraining plus unsupervised motion-word segmentation) with an anchor-guided cluster-and-pseudo-label pipeline transferred from image-domain few-shot recognition. The framework is designed so that evolving evaluation criteria can be absorbed by updating only a lightweight semantic layer instead of re-annotating and re-training the whole system.

> ⚠️ 措辞纪律（v0.3）：此处用能力性主张 "is designed so that ... can be absorbed"——结果性动词 "require" 在 C1 证据落地前禁用（claim ledger 门禁）。

**三支柱自检**（galaxy Narrative Principle）：
- **The What**: 物理-语义解耦 + 骨架域小样本伪标签迁移（W1 核验：组合首次性初步成立）。
- **The Why**: P0.1 实测证据链（kNN 2.51× 随机基线）+ 后续 P0.2-P0.5 实验矩阵；诚实边界：投稿前 Scholar 终审保留。
- **The So What**: 工作犬行业标注成本高企（>50% 训练淘汰率、单犬 >$12k，[Science 特稿]）；业务评估标准持续演化是行业常态而非例外。

---

## 1. 标题（沿用定位文档 v1.1）

*A Physics-Semantics Decoupled Framework for Low-Resource Animal Behavior Recognition under Evolving Evaluation Criteria*

---

## 2. Claims–Evidence 矩阵（大纲的脊柱，paper-plan 规范）

| # | Claim（主张） | Evidence（证据） | 状态 | 落点 |
|---|--------------|------------------|------|------|
| C1 | 物理-语义解耦使评估标准演化时仅需更新语义层，成本远低于全管线重训 | 解耦切换实验（标注单元数 + 墙钟时间对比非解耦基线） | ⏳ 待 P0.5 | §3.4 + §5 |
| C2 | **首次**将图像域小样本伪标签组合（锚点种子 + 原型聚类 + 迭代自训练）迁移到时序骨架识别 | W1 排查矩阵：10 组查询 + 14 项候选零占坑（`NOVELTY_CHECK_YAOQING_JIA.md` §2-§3）；边界：arXiv/Scholar 人工终审投稿前执行 | ✅ 可写（须带边界声明） | §1 贡献点 + §2.3 |
| C3 | 无任何行为标注的动物骨架流上，自监督预训练即可产生可区分的动力学表征 | P0.1 实测：kNN top-1 **20.89%** vs 随机 8.33%（**2.51×**，5-fold CV，dog-ID 代理探针口径已披露于 `reports/p01-aimclr-2026-08-23.md` §2） | ✅ 已可写 | §3.2 + §4.3 |
| C4 | 无监督运动词量化能在连续骨架流上切出与真实边界对齐的行为单元 | SMQ 边界 IoU vs 滑动窗口基线 | ⏳ 待 P0.2 | §3.2 + §4.3 |
| C5 | 少量规则引擎种子锚点即可经聚类伪标签迭代扩展语义覆盖 | 聚类纯度 / 伪标签置信度分布随迭代变化曲线 | ⏳ 待 P0.3 | §3.3 |
| C6 | 半监督自训练以 ≤20% 标注量逼近全监督水平（人类域参照：TCL 10% 标注达 82.7%，全监督 88.6%） | 三层口径主结果表 | ⏳ 待 P0.4 | §4.3 |
| C7 | 不确定性采样主动学习使 100–200 片段人工预算达到 22 类 ≥85% | 主动学习效率曲线（vs 随机采样） | ⏳ 待 P0.5 | §4.3 |
| C8 | （工程副产物）官方 AimCLR 初始化在本数据域诱发表征坍缩，跳过该初始化即恢复收敛 | E1-E7 诊断实验链（`reports/p01-aimclr-2026-08-23.md` §4） | ✅ 已可写（作复现性脚注或附录） | §3.2 脚注 |

> 引用纪律：Evidence 列中一切对外数字必须能溯源到 `reports/` 归档文件；池外引用一律 `[CITATION-NEEDED]`。

---

## 3. 故事线（Narrative Arc）

1. **钩子**：动物行为识别有真实产业痛点（工作犬培养淘汰率过半、成本高昂），但监督学习依赖大规模标注，而行为学标注恰恰最贵。
2. **缺口**：(i) 动物域骨架识别缺乏统一低资源框架；(ii) 自监督/半监督机器全部集中在人类域 NTU 协议；(iii) 业务评估标准持续演化，固定标签集的方法每次都要推倒重来。
3. **方案**：把"骨架怎么动"（物理层，免标注）与"行为叫什么"（语义层，轻标注）解耦；物理层用自监督预训练 + 无监督分割吃透无标签流，语义层用锚点引导的伪标签迭代 + 半监督自训练吃小预算。
4. **证据**：三层数据口径（合成 / 公开真实 / 真实 K9）下系统验证 C1-C8。
5. **回响**：评估标准演化实验证明只需换语义层——标注经济性与可持续性同时解决。

---

## 4. 章节规划（逐节：内容 / 关键 claim / 图表 / 状态）

### Abstract（150–250 词）— ✅ 占位初稿已成 `introduction.md`（数字占位，终稿待回填）
- 四要素齐全（背景/方法/结果/结论）+ **≥3 个数值结果**（[RESULT-1] kNN 倍率已实填、[RESULT-2] 22 类精度、[RESULT-3] 标注预算节省比）。
- 五句公式（Farquhar）：成果一句 → 为什么难 → 怎么做（关键词可检索性）→ 证据 → 最亮眼的数字。
- ❌ 删开头式套话（"Recently, ... has attracted increasing attention" 类）。

### 1. Introduction（~1.5 页）— ✅ 六段式初稿已成 `introduction.md`（数字占位，终稿待 P0 数据）
- 贡献 bullets ×4（每条 ≤2 行）：
  1. 提出 物理-语义解耦框架，形式化"评估标准演化"并给出解耦机制（C1）；
  2. 首次将图像域锚点-聚类-伪标签组合迁移到时序骨架域（C2，附核验边界）；
  3. 在公开真实动物骨架数据上验证免标注预训练与无监督分割的有效性（C3/C4）；
  4. 给出三层数据口径下的完整低资源管线与主动学习闭环（C5-C7）。
- Hero figure：fig1 框架总览图，落点 §1 尾（v0.3 定稿），绘制规格见 `figure-specs.md`。
- Methods 必须从第 2-3 页开始（galaxy 红线）。

### 2. Related Work（~1.5 页）— ✅ **已成稿** `related-work.md`
- 主题式组织（禁逐篇罗列），三小节：
  - 2.1 Animal behavior understanding（RGB 路线 vs 姿态路线 vs 传感路线）
  - 2.2 Skeleton-based self-supervised and unsupervised learning（人类域集中现象）
  - 2.3 Semi-/few-shot learning and pseudo-labeling（三近邻差距表 tab1 收尾）
- 每小节以一句差距句（gap sentence）收尾，直接铺垫本文动机。
- 关键 claim 支撑：C2 的定位合法性在此建立。

### 3. Method（~4 页）— ✅ **框架已成稿** `method.md`（数字留占位符）
- 3.1 Problem formulation and overview（符号表 + 解耦接口定义）
- 3.2 Physics layer：3.2.1 自监督预训练（AimCLR 适配 + 坍缩修复工程注记 C8）；3.2.2 无监督时序分割（运动词量化）
- 3.3 Semantic layer：3.3.1 锚点学习；3.3.2 原型聚类 + 伪标签迭代（Algorithm 1 伪代码）；3.3.3 半监督自训练与主动学习闭环
- 3.4 Decoupling mechanism：评估标准演化的形式化（taxonomy T → T′ 只触发语义层重训）+ 成本度量定义
- 图表：fig1 框架总览、fig2 语义层迭代闭环细节、Algorithm 1。

### 4. Experiments（~3 页）— ⏳ **骨架已成稿** `experiment-skeleton.md`（数字全占位）
- 4.1 Datasets and the three-tier protocol（实测值可直接写：InterPet4D 226 npz/225 有效、AK 犬科 329 视频/34,772 帧行、APTv2 83,304 文件）
- 4.2 Implementation details（RTX 5060 8GB 单卡、seed 策略、5-fold、误差棒）
- 4.3 Main results（tab2 三层主表 + 各 claim 对应实验）
- 4.4 Comparison studies（组件级基线对比）
- 每个实验先写一句"This experiment tests whether [claim]"（galaxy Experiments 规范）。

### 5. Ablation and Analysis（~2 页）— ⏳ 全部待数据，骨架进 experiment-skeleton.md
- 消融矩阵：预训练开关 / 分割开关 / 锚点数扫描 / 伪标签迭代轮数 / 主动学习策略。
- 敏感性分析：种子比例、聚类数 K、置信度阈值。

### 6. Conclusion and Limitations（~0.5 页）— ✅ 骨架已成 `conclusion-limitations.md`（含 4 条 Limitations + rebuttal 预案）
- Limitations 独立成段（诚实原则）：代理探针口径、单一物种家族、Scholar 终审未做、消费级算力规模。

---

## 5. 图表规划（Figure & Table Plan，paper-plan Phase B 审计通过）

| ID | 类型 | 内容 | 落点 | 状态 |
|----|------|------|------|------|
| fig1 | 框架总览图（hero） | 物理-语义双层架构 + 数据流 | §1 尾（v0.3 定稿） | 结构已在 method.md 文字化，绘图待排期 |
| fig2 | 架构细节图 | 锚点-聚类-伪标签迭代闭环 | §3.3 | 待画 |
| fig3 | 定性可视化 | SMQ 分割边界 vs GT | §4.3 | ⏳ 待 P0.2 |
| fig4 | 效率曲线 | 主动学习标注预算 vs 精度 | §4.3 | ⏳ 待 P0.5 |
| tab1 | 对照表 | 三近邻差距（MAC-Learning / Skeleton-to-Image / TP-CanineNet） | §2.3 | ✅ related-work.md 已含 |
| tab2 | 主结果表 | 三层口径 × 方法 × 指标 | §4.3 | ⏳ 骨架已建（experiment-skeleton.md） |
| tab3 | 消融表 | 组件开关矩阵 | §5 | ⏳ 待数据 |

> 密度自检（Phase C）：Method 有架构图 + 伪代码 ✅；每个实验节有图表位 ✅；无 >3 页纯文字节 ✅；Intro 有 hero figure 位 ✅。感知类定性对比（fig3）已规划——分割任务的可视门面图。

---

## 6. 引用计划（Citation Plan，逐节）

| 章节 | 引用条目（均溯源 `dev-docs/research/RESEARCH_LITERATURE.md` 17 篇池 / `NOVELTY_CHECK_YAOQING_JIA.md`） | 出处标识 | 验证状态 |
|------|------|------|------|
| §1 Intro | 工作犬产业背景（Science 特稿） | science.org/content/article/can-science-build-better-working-dog | ✅ 池内（注意：新闻特稿，正文措辞用 report/feature 而非 study） |
| §1 Intro | IMU+ML 工作犬姿势评估（部署佐证） | PLOS ONE 2023, PMC10284380 | ✅ 池内 |
| §2.1 | TP-CanineNet | MDPI Animals 2025（无代码） | ✅ 池内 |
| §2.1 | BCST-GCN | DOI 10.3389/fvets.2026.1782396 | ✅ 池内 |
| §2.1 | ASBAR | github.com/MitchFuchs/asbar（eLife 2024） | ✅ 池内 |
| §2.1 | 层次化犬行为（单目 3D 姿态） | NeurIPS 2025（具体条目名待补全） | ⚠️ 池内有索引、缺完整题录 → 补全前标 [CITATION-NEEDED] |
| §2.1 | DeepLabCut / SLEAP | DOI 10.1038/s41593-018-0209-y / 10.1038/s41592-022-01426-1 | ✅ 池内 |
| §2.1 | YOLO-PetX | IEEE CEECT 2025 | ✅ 池内（会议论文，题录待补全） |
| §2.2 | ST-GCN | arXiv 1801.07455（AAAI 2018） | ✅ 池内 |
| §2.2 | PoseC3D | CVPR 2022, OpenMMLab | ✅ 池内（DOI 待补） |
| §2.2 | AimCLR | AAAI 2022；repo Levigty/AimCLR（W1 核验在案） | ✅ 双源 |
| §2.2 | AimCLR++ | Pattern Recognition 2024；repo Levigty/AimCLR-v2（W1 附带发现） | ✅ 双源 |
| §2.2 | SMQ（Skeleton Motion Words） | ICCV 2025（arXiv 号待补） | ⚠️ [CITATION-NEEDED: arXiv ID] |
| §2.2 | VideoMamba / Mamba-MSQNet | ECCV 2024 / Ecological Informatics 2024 | ⚠️ 池内有索引；Mamba-MSQNet 缺完整题录（RW 内嵌 [CITATION-NEEDED]） |
| §2.2/§2.3 | 跨视角 SSL 代表工作 / Momentum-Contrastive Teacher (TIP 2025) / GRA | — | ❌ [CITATION-NEEDED]（RW 内嵌标记，随文献终审补齐） |
| §2.3 | MAC-Learning | TPAMI 2022；repo 1xbq1/MAC-Learning | ✅ 双源 |
| §2.3 | Skeleton-to-Image Encoding | arXiv 2603.05963 | ✅ 池内 |
| §2.3 | TCL | CVPR 2021 | ⚠️ [CITATION-NEEDED: DOI/repo] |
| §2.3 | DINO（VFM 代表） | — | ❌ [CITATION-NEEDED]（池外） |
| §4.1 | **InterPet4D 数据集论文** | — | ❌ [CITATION-NEEDED]（池外；用了数据集必须引出处，PR 硬要求） |
| §4.1 | **Animal Kingdom 数据集论文** | — | ❌ [CITATION-NEEDED]（池外，同上） |
| §4.1 | **APTv2 数据集论文** | — | ❌ [CITATION-NEEDED]（池外，同上） |

> 铁律执行情况：池外或题录不全者一律 `[CITATION-NEEDED]` 并在文中显式标记；绝不凭记忆生成题录。
> ⚠️ 数据集引用为收录硬门槛（v0.2 评审新增）：三个数据集的原始论文题录必须在文献终审窗口补齐，否则直接进 Major Revision。

---

## 7. 页面预算（PR 双栏终稿估算，供排版期校准）

Abstract 0.2 ｜ Intro 1.5 ｜ RW 1.5 ｜ Method 4 ｜ Experiments 3 ｜ Ablation 2 ｜ Conclusion+Limitations 0.5 ≈ **12.7 页正文**（参考文献不计页）。PR 无硬性页限，此预算用于控制篇幅均衡。

---

## 8. 审稿风险登记册与收录检查单（v0.2 新增，五视角对抗评审产物）

### 8.1 风险登记册（按严重度排序；🔴 未消除前不投稿）

| # | 级别 | 风险 | 缓解动作 | 状态 |
|---|------|------|---------|------|
| R1 | 🔴 CRITICAL | **行为级证据为零**：唯一实测数字是 dog-ID 代理探针，审稿人将质疑"承诺行为识别却无行为识别证据" | C3 措辞永久锁死在"表征区分度"层面；C4-C7 行为级主张全部显式挂 PENDING；P0.4/P0.5 是投稿硬前置 | ⏳ 待实验 |
| R2 | 🔴 CRITICAL | **标题过度主张风险**：副句 "under Evolving Evaluation Criteria" 的唯一证据 E6 待做 | E6 设计已在 §3.4 形式化；预案：E6 若弱则降级标题为 "...for Low-Resource Animal Behavior Recognition"（投稿前用户裁决） | ⏳ 待实验 |
| R3 | 🟠 MAJOR | **incremental 指控**："已知组件组合 + 换域 = 增量工作" | 解耦机制列第一贡献；method.md 补迁移非平凡性论证段；related-work.md 三近邻表量化差异 | ✅ 写作侧已加固 |
| R4 | 🟠 MAJOR | **重实现正确性存疑**：AimCLR/SMQ/TCL 均为本仓适配实现，审稿人会要求验证实现等价性 | ✅ 用户已批准纳入（ADR 0002）；⚠️ 前置=NTU60 骨架数据获取（本仓现无），P0.2 释放 GPU 后独立小窗口执行 | 📋 已批准待排程 |
| R5 | 🟠 MAJOR | **数据集论文未引**：InterPet4D/AK/APTv2 引用缺失（PR 投稿硬要求） | 引用计划 §6 已补三条 [CITATION-NEEDED]；文献终审窗口必须补齐 | ⏳ 待终审 |
| R6 | 🟡 MINOR | 统计严谨性质疑 | experiment-skeleton 已补统计协议节 | ✅ 协议已立 |
| R7 | 🟡 MINOR | 数据许可与伦理声明（PR 投稿系统必填） | brief §8 许可终审项覆盖；投稿前补 Data Availability + Ethics 声明 | ⏳ 已跟踪 |
| R8 | 🟠 MAJOR | **种子噪声放大**：规则引擎粗标带噪，伪标签迭代可能放大错误 | method §3.3.2 已立"高召回低精度先验"设计决策；消融表已加噪声注入实验行（10/20/30%） | ✅ 设计+验证已闭环 |
| R9 | 🟠 MAJOR | **类别长尾分布**无应对表述——动物行为极不平衡是领域常识 | method §3.3.2 补 frequency-aware margin 设计决策；敏感性扫描加处理方式对比 | ✅ 已闭环 |
| R10 | 🟠 MAJOR | **E6 演化场景真实性**：作者自造 Y′ 会被批稻草人 | E6 已加 Y′ 来源要求（真实业务依据或基准化协议）+ matched accuracy 判据固定条款 | ✅ 协议已立，Y′ 选型待 P0.5 前 |

### 8.2 PR 收录检查单映射

| PR 典型录用要件 | 本文现状 |
|----------------|---------|
| 方法论新颖性（非纯应用） | ✅ 解耦框架 + 迁移组合（W1 核验）；写作侧已防 incremental 指控 |
| 广泛充分的实验 | ⏳ P0.2-P0.5 全部待做——**当前最大短板，投稿前置** |
| 与 SOTA 对比 | ⏳ 对比研究骨架已建；NTU 复现行待用户决策 |
| 统计严谨（误差棒+显著性检验） | ✅ 协议已立，待数据 |
| 可复现性（代码/超参公开） | ✅ 一条命令复现链已有（P0.1 实测）；开源计划见定位文档 §6 |
| 诚实局限声明 | ✅ Limitations 独立成段设计 |
| 图表质量规范 | ✅ 规范节已立（白底/≤6色/色盲安全/矢量图） |

## 9. 本窗口完成度声明（诚实版）

- ✅ 本大纲（含 Claims-Evidence 矩阵、图表规划、引用计划）
- ✅ `related-work.md` 英文初稿 1074 词（经两轮对抗评审，见 `review-log.md`）
- ✅ `method.md` 英文框架（数字留占位符）
- ✅ `experiment-skeleton.md` 三层口径骨架（含统计协议与图表规范）
- ✅ 增量二：`introduction.md`（Abstract 占位稿 + Intro 六段式）、`conclusion-limitations.md`、`figure-specs.md`
- 🚫 不做：实验数字填充（等 P0.2-P0.5）、Introduction/Abstract 终稿回填、投稿格式排版、矢量终图绘制

## 10. 方法论适配记录（工具集 → PR 英文期刊裁剪）

| 工具集原规则 | 本仓适配 | 依据 |
|------|------|------|
| 摘要四要素 + ≥3 数值结果 | 直接适用，数值暂用占位符计数 | 交接文档 §4 |
| 图表白底浅灰网格 / ≤6 色 / 低饱和 / 避红绿 | 写入 experiment-skeleton.md 图表规范节，画图阶段生效 | 同上 |
| 中文短句占比 60-65% | 英文化为句均长 ≤25 词、被动语态慎用、避免 AI 腔 | 同上 |
| FIGURE_MANIFEST 机器对账区块 | **不采用**——该区块服务于工具集内部 paper-figure 流水线，本仓无此下游；改用 §5 人类可读图表规划表 | 交接文档 §2「技能文档若与列名有出入……记录实际所用」 |
| sci-literature-review 强制 AI 生成示意图 | **不采用**——期刊 Related Work 章节不要求独立配图；检索环节亦不适用（AGENTS.md 禁 WebSearch，文献池已由 W1 GitHub-First 建立） | 同上 + AGENTS.md 硬规则 1 |

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 窗口建纲：故事线 + C1-C8 矩阵 + 逐节规划 + 图表/引用计划 |
| v0.2 | 2026-08-23 | 五视角对抗评审：新增 §8 风险登记册（R1-R7，含两条 CRITICAL）+ PR 收录检查单；引用计划补三个数据集论文 [CITATION-NEEDED]；章节重编号 |
| v0.3 | 2026-08-23 | 第二轮对抗评审：§0 贡献句降级为能力性主张 + 风险登记册增 R8-R10（种子噪声/类别不平衡/E6 场景真实性）+ fig1 落点决策闭合；评审全程记录见 `review-log.md` |
| v0.4 | 2026-08-23 | 增量二：Abstract/Introduction/Conclusion-Limitations 占位初稿（introduction.md / conclusion-limitations.md）+ fig1/fig2 绘制规格（figure-specs.md）；§4/§5/§9 状态同步。W5 写作侧可做项至此清空，剩余全部为 P0 数据依赖 |
