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
| C1 | 物理-语义解耦使评估标准演化时仅需更新语义层，成本远低于全管线重训 | C1 解耦成本实验（W19+relay Q2）：full 档墙钟 **6.07×**（31.1s vs 188.7s 干净 GPU，同 seed 配对最小 4.00×）、small 档 **7.32×**——论文措辞保守区间 **≥3×**；精度统计等效（full −0.91pp / small +2.27pp，均 <2.3pp）；标注单元数两臂打平（如实）——`reports/c1-decouple-cost-full-2026-08-25.json` + `reports/c1-decouple-cost-2026-08-24.md` §2+§9 | 🟢 合成层双档实证闭合（2026-08-25 W34 回写+W36 矩阵同步）；标题保留与否归用户终裁 | §3.4 + §5 |
| C2 | **首次**将图像域小样本伪标签组合（锚点种子 + 原型聚类 + 迭代自训练）迁移到时序骨架识别 | W1 排查矩阵：10 组查询 + 14 项候选零占坑（`NOVELTY_CHECK_YAOQING_JIA.md` §2-§3）；边界：arXiv/Scholar 人工终审投稿前执行 | ✅ 可写（须带边界声明） | §1 贡献点 + §2.3 |
| C3 | 无任何行为标注的动物骨架流上，自监督预训练即可产生可区分的动力学表征 | P0.1 实测：kNN top-1 **20.89%** vs 随机 8.33%（**2.51×**，5-fold CV，dog-ID 代理探针口径已披露于 `reports/p01-aimclr-2026-08-23.md` §2） | ✅ 已可写 | §3.2 + §4.3 |
| C4 | 无监督运动词量化能在连续骨架流上切出与真实边界对齐的行为单元 | SMQ 边界 IoU **0.458±0.049** vs 等段数随机切分 null ≈0.30/实测 0.323±0.022（seeds 伪 GT 协议）；**W38 三臂消融**：SMQ > 均匀滑窗 0.399±0.035（预注册三门全过）> null → 结构化分割不可约简为任意切分；⚠️ 固定网格旁证臂 0.453 与 SMQ 统计等效 + boundary F1 反向，措辞限"优于任意切分+等段数下最优"（`reports/p02-smq-iou-eC-seeds-recheck.json` + `reports/p02-seg-strategy-ablation-2026-08-25.md`） | ✅ 已可写（含消融与双向论证） | §3.2 + §4.3 + §5 |
| C5 | 少量规则引擎种子锚点即可经聚类伪标签迭代扩展语义覆盖 | 聚类纯度 **0.5339** vs 随机基线 0.3306（**1.62×**）；30% 标签噪声仅降 3.1pp；种子比例 25%→100% 增益 ≤1pp（原型早期饱和）（`reports/p03-jia-phasea-results.json`） | ✅ 已可写 | §3.3 |
| C6 | 半监督自训练以 ≤20% 标注量逼近全监督水平（~~人类域参照 TCL 82.7/88.6~~ R8 证伪删除：原文无 NTU 无此数字） | 池精度 **0.691±0.013**（r1 操作点 cov≈35%）+ 首末配对 Δ **+10.69±3.28pp (p=0.030)**（`reports/p04-tcl-results.json`）；**端到端补强（E7+P1.0，2026-09-04，n=10 主口径）**[R16 撤回：真标签头协议错误产物，修正协议下 AK 端到端近随机——负边界，见 skeleton v1.9]：完整管线 13% 标注预算达全监督 ~~94%~~（0.3196±0.0245 vs 0.3393，公开真实层 AK full12），warm vs scratch 端到端 +7.3pp 10-0-0 p=0.002/macro-F1 3.3×（`reports/p10-seedexpansion-2026-09-04.json`；n=3 首轮归档 `reports/p07-endtoend-ak-full12-2026-09-04.json`）——旧"池精度而非端到端"注记由 E7 解除，端到端量化成立；绝对天花板 ~34% 归数据瓶颈（L7）如实；**P0.8 同协议消融 + P1.0 种子扩容（2026-09-04）**：唯一变量替换 warm-start→通用 SSL（AimCLR@InterPet4D），n=10 下 top-1 全预算**显著**占优（spc2 +5.89pp 10-0-0 p=0.002 / spc4 p=0.016）、macro-F1 持平（p=0.77，n=3 反转消解为种子噪声）——"语义 warm-start 是低预算关键"获同协议显著对照（`reports/p08-aimclr-arm-2026-09-04.md` + `reports/p10-seedexpansion-2026-09-04.md`）；**E7b v2 预注册复现（2026-09-04）**[R16 撤回：修正协议下 v2 各臂近随机]：352 clips 多段层上 warm vs aimclr 双指标 10-0-0 p=0.002 增强复现，且 EP3 判据触发**数据瓶颈成立**（v2 full 37.50%=v1+3.57pp≥+3.0pp 冻结线）——天花板归因升级为预注册检验结论（`reports/p12-akv2-replication-2026-09-04.md`）；**P1.3 微调诊断对照**[R17 修正：33.23 系真标签头产物，对照基准改为修正协议近随机臂；macro-F1"反向"系评估器 bug，修正后双指标同向]：端到端解冻 full 32.99<冻结 37.50（过拟合）/spc2 9.72 vs ~~warm 33.23~~=3.4×——『天花板非冻结所致』防御句入稿（`reports/p13-v2-finetune-2026-09-04.md`）；**E9 NTU 预注册保留率（2026-09-04）**[R16 修正：99.5%→90.6%，+8.1→+1.4pp，GENERALIZES 仍成立]：人体域 10% 标注+selftrain 保留 ~~99.5%~~（74.11 vs 74.45，判据 GENERALIZES，伪标签迭代 +8.1pp；R8 修订：TCL 对比证伪删除，仅比预注册 90% 线+协议依赖披露）——低资源主张跨域成立（`reports/p14-ntu-lowres-2026-09-04.md`） | 🟡 **R16 修正（2026-09-05）**：端到端主张跨域成立（NTU 修正协议保留率 90.6%≥预注册 90% 线），AK 层端到端撤回为负边界（13% 预算近随机，伪标签精度≈0.11）；组件级 E4（0.691 共识口径）与合成层 warm-start（82%）不受影响；原"端到端量化成立（E7）"系真标签头协议错误产物。**P1 标签对齐修正线（2026-09-05，PSD-ALIGN-PREREG-001）NULL**：V2 14.11%（胜 8/10 但未达 20% 线），约束=特征几何非门控，负边界维持，§5 消融入文。**P2 APTv2-DAP 骨干（PSD-DAP-PREREG-001）NULL+灾难遗忘、P3 预算曲线（PSD-BUDGET-PREREG-001）膝点未达**：三杠杆一致否证 AK 端到端可救援 | §4.3 E4+E7/E7b/E9 + §5 P1 |
| C7 | ~~不确定性采样主动学习使 100–200 片段人工预算达到 22 类 ≥85%~~（**该主张经用户裁决①降级，正文与摘要不再声明**）→ **换轨新主张（用户裁决 A，2026-08-25）：warm-start 语义层初始化使小标注预算可用**——合成偏移层 82.0%@20 片段（vs 冷启动同预算 ~7.8%），两轮实验背书（W23 短档 + 冒烟链） | 效率负结果（边界条件，探索性发现）：W14 冷启动熵未占优 + **W23 强打分器下仍负**（随机 3/3 seeds × 3/3 预算点反超 4.2~5.0pp——排除弱打分器混杂）+ relay Q1 extended 复跑方向一致（b=100 随机 +12.3pp / b=200 +7.8pp）三重证据闭环；softmax 全饱和诊断；`reports/w14-p05-al-efficiency-2026-08-24.md` + `reports/w23-p05-al-warmstart-2026-08-25.md` + `reports/p05-al-efficiency-full-2026-08-25.json` | 🔁 已裁决 A（2026-08-25）：E5 叙事换轨"warm-start 可用性"，效率曲线以探索性发现进 §5；类间轮转熵混合策略一句话 future work | §4.3-E5 + §5 |
| C8 | （工程副产物）官方 AimCLR 初始化在本数据域诱发表征坍缩，跳过该初始化即恢复收敛 | E1-E7 诊断实验链（`reports/p01-aimclr-2026-08-23.md` §4） | ✅ 已可写（作复现性脚注或附录） | §3.2 脚注 |

> 引用纪律：Evidence 列中一切对外数字必须能溯源到 `reports/` 归档文件；池外引用一律 `[CITATION-NEEDED]`。

---

## 3. 故事线（Narrative Arc）

1. **钩子**：动物行为识别有真实产业痛点（工作犬培养淘汰率过半、成本高昂），但监督学习依赖大规模标注，而行为学标注恰恰最贵。
2. **缺口**：(i) 动物域骨架识别缺乏统一低资源框架；(ii) 自监督/半监督机器全部集中在人类域 NTU 协议；(iii) 业务评估标准持续演化，固定标签集的方法每次都要推倒重来。
3. **方案**：把"骨架怎么动"（物理层，免标注）与"行为叫什么"（语义层，轻标注）解耦；物理层用自监督预训练 + 无监督分割吃透无标签流，语义层用锚点引导的伪标签迭代 + 半监督自训练吃小预算。
4. **证据**：三层口径中两个已执行层（合成 / 公开真实）+ 人体域验证基准下系统验证 C1-C8（C7 经用户裁决①以探索性发现呈现；C6 端到端结果 tier-dependent：NTU 90.6% 过预注册线、犬科层 13% 预算为负边界——R16 修正）。
5. **回响**：评估标准演化实验证明只需换语义层——标注经济性与可持续性同时解决。

---

## 4. 章节规划（逐节：内容 / 关键 claim / 图表 / 状态）

### Abstract（150–250 词）— ✅ **终稿候选已成（W36 v0.3 合并定稿，见 introduction.md）**
- 四要素齐全 + **3 个数值结果全部落位**：[RESULT-1] kNN 2.51× ✅、[RESULT-2] warm-start 82.0%@20 clips（合成偏移层口径随句，W36 终填）✅、[RESULT-3] 用户裁决候选 C 定稿——解耦墙钟 ≥3× 保守界（B 候选 +10.69pp / A 候选 SMQ 1.53× 否决留痕）✅。
- 五句公式（Farquhar）：成果一句 → 为什么难 → 怎么做（关键词可检索性）→ 证据 → 最亮眼的数字。
- ❌ 删开头式套话（"Recently, ... has attracted increasing attention" 类）。

### 1. Introduction（~1.5 页）— ✅ 六段式初稿已成 `introduction.md`（数字占位，终稿待 P0 数据）
- 贡献 bullets ×4（每条 ≤2 行）：
  1. 提出 物理-语义解耦框架，形式化"评估标准演化"并给出解耦机制（C1）；
  2. 首次将图像域锚点-聚类-伪标签组合迁移到时序骨架域（C2，附核验边界）；
  3. 在公开真实动物骨架数据上验证免标注预训练与无监督分割的有效性（C3/C4）；
  4. 给出三层数据口径下的完整低资源管线（C5-C6）；主动学习以探索性发现形式进 §5 分析节（C7 经用户裁决①降级，2026-08-25）。
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
| fig4 | 效率曲线 | 主动学习标注预算 vs 精度（负结果曲线：熵 vs 随机，合成层短预算档） | §5（C7 裁决①降级后随 E5 迁入分析节） | 🔄 曲线数据已归档（W14 负结果），W22 绘制中；full-budget 待 GPU |
| tab1 | 对照表 | 三近邻差距（MAC-Learning / Skeleton-to-Image / TP-CanineNet） | §2.3 | ✅ related-work.md 已含 |
| tab2 | 主结果表 | 三层口径 × 方法 × 指标 | §4.3 | 🟡 素材就绪：合成层 E1-E4（quickref 复核值）、公开真实层 P0.4 0.691 + Q3c 44.90%（4 类部分口径，逐类分解同框）；真实 K9 列空缺如实标注；成表排版待终稿期 |
| tab3 | 消融表 | 组件开关矩阵 | §5 | ✅ **六行全部映射溯源完成（2026-08-26 W36 收口；同日 W45 预训练行梯度升级）**：P0.3/P0.4/W14/W38 滑窗臂/W39 预训练消融——预训练行已从单点零结果升级为四档梯度叙事（spc5 +21.21pp / spc10 +9.85pp / spc20 −2.27pp 交叉区 / full +0.15pp n.s.，收益随标注资源单调衰减；低资源与饱和档成对呈现、CPU-GPU 禁混排纪律随行）——experiment-skeleton.md §5 |

> 密度自检（Phase C）：Method 有架构图 + 伪代码 ✅；每个实验节有图表位 ✅；无 >3 页纯文字节 ✅；Intro 有 hero figure 位 ✅。感知类定性对比（fig3）已规划——分割任务的可视门面图。

---

## 6. 引用计划（Citation Plan，逐节）

| 章节 | 引用条目（溯源 `dev-docs/research/RESEARCH_LITERATURE.md` / `NOVELTY_CHECK_YAOQING_JIA.md` / 官方仓 / arXiv API） | 出处标识 | 验证状态 |
|------|------|------|------|
| §1 Intro | 工作犬产业背景（Science 特稿） | Grimm, D., "Can science build a better working dog?", Science news feature, science.org/content/article/can-science-build-better-working-dog（2026-02） | ✅ 池内题录齐备（新闻特稿，正文措辞用 report/feature 而非 study） |
| §1 Intro | IMU+ML 工作犬姿势评估（部署佐证） | PLOS ONE 2023, PMC10284380 | ✅ 池内 |
| §2.1 | TP-CanineNet | MDPI Animals 2025（无代码） | ✅ 池内 |
| §2.1 | BCST-GCN | DOI 10.3389/fvets.2026.1782396 | ✅ 池内 |
| §2.1 | ASBAR | github.com/MitchFuchs/asbar（eLife 2024） | ✅ 双源 |
| §2.1 | 层次化犬行为（单目 3D 姿态） | Miyai, Kubo, Saito, Ohno, Kikusui, Nagasawa, Ikeda, "Hierarchical Representation Learning of Dog Behavior via Single-View 3D Pose Estimation", **NeurIPS 2025 Workshop on AI for Animal Communication**, OpenReview EDeOoWN4PT | ✅ 已补全（W17 终审；⚠️ 系 Workshop 论文非主会，正文措辞已按 workshop 书写） |
| §2.1 | DeepLabCut / SLEAP | DOI 10.1038/s41593-018-0209-y / 10.1038/s41592-022-01426-1 | ✅ 池内 |
| §2.1 | YOLO-PetX | 全题名 "YOLO-PetX: Enhanced YOLO-Based Recognition of Abnormal Dog Behaviors in Intelligent Pet Care Applications"，IEEE CEECT 2025（Xplore 上线 2026-02，经 GitHub 全局检索锁定） | ⚠️ 题名/会议双源锁定；作者列表与 DOI 无法经允许渠道获得 → 正文缩窄标记 `[CITATION-NEEDED: authors+DOI]`，Scholar 终审补 |
| §2.2 | ST-GCN | arXiv 1801.07455（AAAI 2018） | ✅ 池内 |
| §2.2 | PoseC3D | Duan, Zhao, Chen, Lin, Dai, "Revisiting Skeleton-based Action Recognition", CVPR 2022 Oral, arXiv 2104.13586 | ✅ 已补全（arXiv API 核验，含作者名单） |
| §2.2 | AimCLR | AAAI 2022；repo Levigty/AimCLR（W1 核验在案） | ✅ 双源 |
| §2.2 | AimCLR++ | Pattern Recognition 2024；repo Levigty/AimCLR-v2（W1 附带发现） | ✅ 双源 |
| §2.2 | SMQ（Skeleton Motion Words） | Gökay, Spurio, Bach, Gall, "Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation", ICCV 2025, pp.12101-12111, arXiv 2508.04513 | ✅ 已补全（external/SMQ 官方仓自带 BibTeX） |
| §2.2 | VideoMamba / Mamba-MSQNet | VideoMamba ECCV 2024；Mamba-MSQNet = Fazzari, Romano, Falchi, Stefanini, "Selective State Models Are What You Need for Animal Action Recognition", Ecological Informatics 2024, art.102955（官方代码仓 edofazza/mamba-msqnet README 指定引用） | ✅ 已补全（官方 BibTeX；池内出处标注经核验正确，另存 MetroAgriFor 2024 会议版线索） |
| §2.2/§2.3 | Momentum Contrastive Teacher (TIP) / GRA | 全题名分别锁定："Momentum Contrastive Teacher for Semi-Supervised Skeleton Action Recognition" IEEE TIP（Xplore 10820022）；"GRA: Graph Representation Alignment for Semi-Supervised Action Recognition" IEEE TNNLS（Xplore 10398229）——均经 firework8/Awesome-Skeleton-based-Action-Recognition 目录回溯验证 | ⚠️ 题名/期刊/链接双源锁定；作者列表待 Scholar 终审 → 正文缩窄标记 `[CITATION-NEEDED: author list]` |
| §2.3 | TCL | **Ankit** Singh, Chakraborty, Varshney, Panda, Feris, Saenko, **Abir** Das, "Semi-Supervised Action Recognition with Temporal Contrastive Learning", CVPR 2021, arXiv 2102.02751 | ✅ 题录真实（方法引用保留 in the spirit of TCL）；❌ 82.7%/88.6%@NTU60 数字 2026-09-04 R8 原文 PDF 证伪（NTU 0 命中），已从论文全链删除；作者名 Jathushan/Amlan 系前轮误修，已回正 |
| §2.3 | 跨视角 SSL 代表工作 | CrosSCLR: "3D Human Action Representation Learning via Cross-View Consistency Pursuit", CVPR 2021, 官方仓 LinguoLi/CrosSCLR（72★） | ✅ 双源（官方仓 + awesome 目录）；作者名单投稿期以原文为准复核 |
| §2.3 | MAC-Learning | TPAMI 2022；repo 1xbq1/MAC-Learning | ✅ 双源 |
| §2.3 | Skeleton-to-Image Encoding | arXiv 2603.05963 | ✅ 池内 |
| §2.3 | DINO（VFM 代表） | Caron et al., "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021, arXiv 2104.14294, repo facebookresearch/dino | ✅ 已补全（官方仓 BibTeX） |
| §4.1 | **InterPet4D 数据集论文** | Peng, Song, Liao, Kitani, Koike, Wu, *InterPet4D: A Multimodal Ego-Centric Dataset of Human–Pet Interactions*, v1, 2026, huggingface.co/datasets/ohicarip/interpet4d（@dataset 条目，CC BY-NC 4.0） | ✅ 已补全（本地数据集目录自带官方 BibTeX；该数据集尚无同行评审论文，引数据集本体即满足 PR 引用要求） |
| §4.1 | **Animal Kingdom 数据集论文** | Ng, Ong, Zheng, Ni, Yeo, Liu, "Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding", CVPR 2022, pp.19023-19034, arXiv 2204.08129 | ✅ 已补全（官方仓 SUTDCV/Animal-Kingdom 自带 BibTeX） |
| §4.1 | **APTv2 数据集论文** | Yang, Deng, Xu, Zhang, "APTv2: Benchmarking Animal Pose Estimation and Tracking with a Large-scale Dataset and Beyond", arXiv 2312.15612（NeurIPS 2022 APT-36K 基准的扩展；官方仓 ViTAE-Transformer/APTv2） | ✅ 已补全（官方仓自带 BibTeX；如需完整谱系可加引 APT-36K NeurIPS 2022 原始论文） |
| §2.1 | **犬行为传感器数据集**（P4 稀缺性证据） | Vehkaoja, Somppi, Kumpulainen, Surakka, Vainio, "Description of Movement Sensor Dataset for Dog Behavior Classification", Data in Brief 40:107822, 2022, DOI 10.1016/j.dib.2022.107822 | ✅ 官方 PMC8777071 页实证（2026-09-05，anysearch extract） |
| §2.1 | **SyDog-Video 合成犬视频**（P4 稀缺性证据） | Shooter, Malleson, Hilton, "SyDog-Video: A Synthetic Dog Video Dataset for Temporal Pose Estimation", IJCV 132(6), 2024, DOI 10.1007/s11263-023-01946-z | ✅ 官方 CVSSP/Springer 页实证（2026-09-05） |

> 铁律执行情况：池外或题录不全者一律 `[CITATION-NEEDED]` 并在文中显式标记；绝不凭记忆生成题录。W17 文献终审（2026-08-24）：11 条完全解决（全部有官方仓 BibTeX / arXiv API 元数据背书），3 条缩窄为作者列表级待补项。
> ⚠️ 数据集引用硬门槛（R5）已于 W17 解除：三条数据集题录均取自官方渠道自带 BibTeX。

---

## 7. 页面预算（PR 双栏终稿估算，供排版期校准）

Abstract 0.2 ｜ Intro 1.5 ｜ RW 1.5 ｜ Method 4 ｜ Experiments 3 ｜ Ablation 2 ｜ Conclusion+Limitations 0.5 ≈ **12.7 页正文**（参考文献不计页）。PR 无硬性页限，此预算用于控制篇幅均衡。

---

## 8. 审稿风险登记册与收录检查单（v0.2 新增，五视角对抗评审产物）

### 8.1 风险登记册（按严重度排序；🔴 未消除前不投稿）

| # | 级别 | 风险 | 缓解动作 | 状态 |
|---|------|------|---------|------|
| R1 | 🔴 CRITICAL | **行为级证据为零**：唯一实测数字是 dog-ID 代理探针，审稿人将质疑"承诺行为识别却无行为识别证据" | C3 措辞永久锁死在"表征区分度"层面；C4-C7 行为级主张全部显式挂 PENDING；P0.4/P0.5 是投稿硬前置 | ⏳ 待实验 |
| R2 | 🔴 CRITICAL | **标题过度主张风险**：副句 "under Evolving Evaluation Criteria" 的唯一证据 E6 待做 | E6 设计已在 §3.4 形式化；预案：E6 若弱则降级标题为 "...for Low-Resource Animal Behavior Recognition"（投稿前用户裁决）；**C1 双档证据齐备**：合成层 small 档（7.32×，保守 ≥3×）+ **full 档确认（6.07×，同 seed 配对最小 4.00×，`reports/c1-decouple-cost-full-2026-08-25.json`）**；精度两档统计等效（full −0.91pp / small +2.27pp，均 <2.3pp）——成本维度实证闭合，标题保留与否归用户终裁 | 🟢 成本维度 full 确认（2026-08-25 W34 回写）；标题终裁待用户 |
| R3 | 🟠 MAJOR | **incremental 指控**："已知组件组合 + 换域 = 增量工作" | 解耦机制列第一贡献；method.md 补迁移非平凡性论证段；related-work.md 三近邻表量化差异 | ✅ 写作侧已加固 |
| R4 | 🟠 MAJOR | **重实现正确性存疑**：AimCLR/SMQ/TCL 均为本仓适配实现，审稿人会要求验证实现等价性 | ✅ 用户已批准纳入（ADR 0002）；⚠️ 前置=NTU60 骨架数据获取（本仓现无），P0.2 释放 GPU 后独立小窗口执行 | ✅ **PASS（2026-09-02 协调者收编）**——NTU60 xsub 三流全交付：joint 74.30 / bone 71.51 / motion 67.84，3s 融合 **top1=77.97%（top5 95.78%, n=16487）≥ 预注册线 77.18%**，与论文正文 78.9% 差 0.93pp 容差内——本仓适配实现等价性成立；证据 `reports/ntu-phaseB-3s-ensemble.json`，回写 experiment-skeleton.md §4.4 |
| R5 | 🟠 MAJOR | **数据集论文未引**：InterPet4D/AK/APTv2 引用缺失（PR 投稿硬要求） | 引用计划 §6 已补三条 [CITATION-NEEDED]；文献终审窗口必须补齐 | ✅ W17 已解除（2026-08-24，三条题录取自官方渠道自带 BibTeX） |
| R6 | 🟡 MINOR | 统计严谨性质疑 | experiment-skeleton 已补统计协议节 | ✅ 协议已立 |
| R7 | 🟡 MINOR | 数据许可与伦理声明（PR 投稿系统必填） | brief §8 许可终审项覆盖；投稿前补 Data Availability + Ethics 声明 | ⏳ 已跟踪 |
| R8 | 🟠 MAJOR | **种子噪声放大**：规则引擎粗标带噪，伪标签迭代可能放大错误 | method §3.3.2 已立"高召回低精度先验"设计决策；消融表已加噪声注入实验行（10/20/30%） | ✅ 设计+验证已闭环 |
| R9 | 🟠 MAJOR | **类别长尾分布**无应对表述——动物行为极不平衡是领域常识 | method §3.3.2 补 frequency-aware margin 设计决策；敏感性扫描加处理方式对比 | ✅ 已闭环 |
| R10 | 🟠 MAJOR | **E6 演化场景真实性**：作者自造 Y′ 会被批稻草人 | 双贴合候选已预注册并获方向性认可：K9 报表粒度差（日报粗/考核细）× 本仓 7 类可计算基础；matched accuracy 判据固定条款已立（ADR 0002 v1.1） | 📋 P0.5 前一句话定稿 |
| R11 | 🟠 MAJOR | **AL 负结果冲击"低资源管线完整性"叙事**（W21 新增）：W14 实证冷启动弱打分器场景下熵采样不优于随机（合成层，3/3 seeds 同向反向差距），且合成训 checkpoint 对真实池 softmax 全饱和——C7 作为贡献点面临降级或撤除 | 双向论证已归档（`reports/w14-p05-al-efficiency-2026-08-24.md` §4-§5）；负结果转化为诚实卖点：不确定性采样前提 = 较强打分器 + 域内校准 → 强化"先标注→校准打分器→再选样"渐进标注叙事；**✅ 用户已裁决①（2026-08-25）**：C7 降级"探索性发现"入 §5 分析节、效率主张正文与摘要禁用；warm-start 正证据候选归 W23（A2 裁决） | ✅ 裁决①缓解落地；W23 产证可经用户再裁升级 |

### 8.2 PR 收录检查单映射

| PR 典型录用要件 | 本文现状 |
|----------------|---------|
| 方法论新颖性（非纯应用） | ✅ 解耦框架 + 迁移组合（W1 核验）；写作侧已防 incremental 指控 |
| 广泛充分的实验 | ⏳ P0.2-P0.5 全部待做——**当前最大短板，投稿前置** |
| 与 SOTA 对比 | 🟡 对比研究骨架已建；**NTU 复现行 ✅ 完成（2026-09-02，三流融合 77.97% PASS）**——公开基准层等价性证据入册 §4.4 |
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
| v0.5 | 2026-08-24 | W17 文献终审：§6 全表 11 条题录完全解决（三数据集/SMQ/TCL/DINO/PoseC3D/犬行为 workshop/CrosSCLR/Mamba-MSQNet/Science 特稿），3 条缩窄为作者列表级待补（YOLO-PetX/TIP Teacher/GRA）；R5 解除；正文内嵌标记同步（related-work/introduction/method）；溯源记录见 review-log.md 终审节 |
| v0.6 | 2026-08-24 | W21 矩阵同步：C1 ⏳→🟡（合成层 small 档实证，full 待确认）；C7 保持 ⏳ + 负结果注记 [PENDING-用户措辞裁决]；R2 🔴→🟡（合成层证据入册）；新增 R11（AL 负结果对管线完整性叙事的影响，待用户裁决 C7 措辞）；fig4/tab3 状态行同步 |
| v0.7 | 2026-08-25 | 用户裁决 C7 选项①落地：矩阵 C7 行标 🔻 降级（落点 §4.3→§5）；§3 故事线与 §4.1 Intro 贡献 bullet 4 计划缩窄（主动学习转探索性发现）；摘要 [RESULT-3] 标注失据待重选源；fig4 落点迁 §5；R11 状态 ✅ 缓解落地 |
| v0.8 | 2026-08-25 | **W36 终稿同步轮**：矩阵 C1 🟡→🟢（full 档 6.07× 闭合）；C4 ⏳→✅（IoU 0.458±0.049 + W34 勘误"非滑窗基线"入 Evidence）；C5 ⏳→✅（purity 0.5339/1.62×/噪声消融）；C6 ⏳→🟡（0.691±0.013 数字就绪、"逼近全监督"措辞按池精度口径收窄留用户终裁）；C7 Evidence 增 relay Q1 extended 三重证据闭环；Abstract 规划行改终稿候选已成（RESULT-2=82.0%@20、RESULT-3=候选 C 定稿）；tab2 ⏳→🟡（公开真实列 Q3c 44.90% 落地）、tab3 注记刷新 W38/W39；引用计划三条作者列表级待补项仍留文献窗（YOLO-PetX 关闭 / GRA 全解决 / MCT 待作者——W32 战果，标记更新归协调者或文献窗） |
| v0.9 | 2026-08-26 | **W36 二轮同步**（重基 master 收编 W33/W38 并行成果）：C4 Evidence 增 W38 三臂消融（SMQ>均匀滑窗>null 预注册三门全过 + 网格平价/F1 反向双披露，措辞边界随行）；tab3 状态改五行已映射/一行执行中；R2 行 🟢 维持（W34 已判） |
| v1.0 | 2026-08-26 | **W36 三轮收口**（吸收 W39 收编）：tab3 ✅ 六行全部映射溯源完成——W39 预训练消融零结果如实入表（Δ=+0.15pp n.s.，预训练价值锚定 kNN 表征证据）；消融表零 PENDING；剩余待办仅 NTU 三流链二次收编与 round2 复核 |
| v1.1 | 2026-08-27 | 协调者移交项同步（W45 移交清单闭合）：tab3 状态行并入四档梯度叙事（原"零结果如实入表"描述已不完整，skeleton 与 conclusion-limitations v0.6 均已含梯度叙事）；其余矩阵/风险册零变更 |
| v1.2 | 2026-09-02 | **三流收官收编轮（协调者，NTU PhaseB 全链 ALL_DONE）**：R4 🟠→✅（NTU60 xsub 三流融合 77.97% ≥ 77.18% 预注册线 PASS——joint 74.30 / bone 71.51 / motion 67.84，top5 95.78%, n=16487，与论文正文 78.9% 差 0.93pp 容差内）；§8.2 检查单"与 SOTA 对比"行 ⏳→🟡（NTU 复现行完成）；experiment-skeleton.md §4.4 终判回写 + v1.2；review-log.md 投稿就绪门 R4 条解除；证据 `reports/ntu-phaseB-3s-ensemble.json` |
| v1.3 | 2026-09-04 | **端到端补强 + 同协议消融 + K9 预注册轮**：C6 行并入 E7 端到端量化（13% 预算达全监督 91%；旧池精度口径保留意见解除）与 P0.8 同协议方法对照（warm vs AimCLR：top-1 全预算占优/macro-F1 混合如实/n=3 未达显著）；tab3 六行→七行（−语义warm-start 行入表，skeleton v1.3）；K9 真实域试点结案为数据物理不存在（k9 仓 ADR 0008 v1.7 零标注废弃 + 生成脚本已删致归一化不可核实，无监督探针亦放弃）→ 预注册协议 `docs/paper/k9-pilot-preregistration.md`（PSD-K9-PREREG-001）+ §4.1 引用句装配；DA 审稿快照 commit 哈希→不可变 tag `review-snapshot`（消除自引用死锁） |
| v1.4 | 2026-09-04 | **P1.0 种子扩容轮**：E7/P0.8 关键对比 n=3→10 + 配对 Wilcoxon——warm vs AimCLR top-1 全预算显著（p=0.002/0.016）、macro-F1 持平（n=3 反转消解）、warm vs scratch 双指标显著；论文主数字切 n=10（91%→**94%**，30.95→31.96±2.45），Abstract/Intro/§4.3/tab2/tab3/§4.4 六处联动；C6 行 Evidence 同步；"n=3 未显著"审稿风险解除 |
| v1.5 | 2026-09-04 | **E7b AK v2 预注册复现轮**：天花板归因升级为预注册检验——PSD-AKV2-PREREG-001 构建前冻结，多段重提取 352 clips（8 类空间披露）；**EP3 触发 DATA_BOTTLENECK_CONFIRMED（v2 full 37.50%=v1+3.57pp≥+3.0pp 冻结线）**；EP2 复现增强（warm spc2 33.23=88.6% 保留@6% 预算；warm vs AimCLR 双指标 10-0-0 p=0.002，v1 macro-F1 持平=单片段伪影）；§4.1/§4.3 E7b/tab2 v2 行装配；v1 数字零替换；C6 Evidence 追加 |
| v1.6 | 2026-09-04 | **P1.3 端到端微调诊断对照轮**：v2 解冻骨干端到端反低于冻结头（32.99 vs 37.50，过拟合吞 pretext 先验）、16 段端到端坍缩 9.72 vs warm-start 33.23=3.4×——『绝对精度不刷』从口径说明升级为带数字的设计反证，装配 §4.3 E7b 段末（control 措辞）；C6 Evidence 追加 |
| v1.7 | 2026-09-04 | **E9 NTU 低资源保留率轮**：预注册协议 PSD-NTU-PREREG-001 实验前冻结，10%+selftrain 保留率 99.5% 触发 GENERALIZES（≥90% 线），超 TCL 发表保留率 93.3%（仅比保留轴）；§4.4 E9 段+tab2 行+Intro 半句装配；C6 Evidence 追加——低资源主张从动物域升级为跨域成立 |
| v2.1 | 2026-09-05 | **P4 调研 + 选项2 动机重写**：crawl4ai+anysearch 穷尽公开检索确认无"真实工作犬+骨架+逐段行为标签"数据集（PawCraft 证伪；Kaggle 犬行为集实为 Vehkaoja 传感器数据；SyDog 合成；Ultralytics Dog-Pose 仅图像）；§2.1 新增"Public data landscape"段把负边界重构为"数据稀缺=论文前提"的实证，§1 加稀缺性从句；新增 2 条官方页实证引用（vehkaoja2022dogsensor/shooter2024sydogvideo）；零实验数字改动。 |
| v2.0 | 2026-09-05 | **R17 回归审计同步**：C6 Evidence 列逐段加 [R16 撤回/修正]/[R17 修正] 行内标记（94%/99.5%/EP2 增强/EP3+P1.3 macro-F1 反向均系协议错误或评估器 bug 产物）；§3 故事线 #4 改 tier-dependent 表述。 |
| v1.9 | 2026-09-05 | **R16 端到端协议诚信修正轮**：对抗审稿实锤端到端族最终头消费池真标签+oracle 停止（"94%@13%/99.5%@10%"与实现不符）；修正协议重跑后 C6 行改判——跨域 NTU 90.6% 过预注册线存活、AK 层转负边界发现；E4 补 Holm（0.090）；详见 experiment-skeleton v1.9 与 review-log R16 | C6 行改判 |
| v1.8 | 2026-09-04 | **R8 诚信修复轮**：TCL 82.7/88.6/NTU60 原文 PDF 证伪（NTU 0 命中）→ 正文/摘要/outline/skeleton/method 全链删除，bib 作者回正（Ankit/Abir）；E7b 8 类同空间对照修复类别空间混淆（+11.54pp 同空间差）；E9 删 30× 不可解释数字、8.0→8.1、删 TCL 对比、补协议依赖/单子集/适配披露；Y_CKPT 泄漏披露补全；微调句双指标限定；tab3/§5/Abstract/Intro 措辞级 8 处；AimCLR++ 77.2→80.9、NTU60 补引 Shahroudy、aimclrpp 题名修正；两协议追加 dated 修订节。恶意审稿判定 Major Revision→修复闭环 |
