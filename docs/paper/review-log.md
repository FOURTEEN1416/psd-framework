# 论文初稿对抗评审记录（Review Log）

> Owner: `docs/paper/review-log.md` · W5 窗口 · 性质: auto-review-loop 的 REVIEW_DOC 等价物（工具集外置脚本不可执行，按其 fallback 条款人工扮演审稿面板）
> 方法论: ars-academic-paper-reviewer 五视角面板 + ars-adversarial-reviewer 强制发现制 + galaxy-paper-self-review claim audit
> 铁律: 每轮每人格至少一条发现，禁止 LGTM；发现必须含「哪里错+为什么+最小修复」

---

## Round 1（2026-08-23 · 五视角面板）

**裁决**: 骨架达到 PR 投稿预备水平；两条 CRITICAL 均为实验依赖，写作侧问题全部闭环。

| 视角 | 发现 | 级别 | 处置 |
|------|------|------|------|
| 😈魔鬼代言人 | 唯一实测数字是 dog-ID 代理探针——零行为级证据 | CRITICAL | 登记册 R1；C3 措辞锁死表征层 |
| 😈魔鬼代言人 | 标题副句 "Evolving Evaluation Criteria" 证据全待做 | CRITICAL | 登记册 R2；预案降级标题 |
| 😈魔鬼代言人 | "组合迁移"必被指控 incremental | MAJOR | method §3.3 非平凡性段 + 三近邻表 |
| 📐方法学 | 无统计显著性检验协议 | MAJOR | experiment-skeleton 统计协议节 |
| 🌏领域 | **三个数据集论文均未列引用** | MAJOR | 引用计划补三条 [CITATION-NEEDED] |
| 🌏领域 | SSL 谱系只写 AimCLR 一支 | MAJOR | §2.2 谱系承认句 |
| 🎯主编 | 缺 NTU 复现正确性验证行 | MINOR | 对比研究【需用户决策】 |

**修复提交**: `a831b8a`

---

## Round 2（2026-08-23 · 三敌意人格强制发现）

**裁决**: **CONCERNS**（无新 CRITICAL；5 WARNING 全部当场闭环）

### 🔨方法论破坏者（"我要在审稿现场击沉这套实验逻辑"）
1. ⚠️ E6 的 Y′ 演化场景来源未定义——作者自造 taxonomy 变更会被批稻草人场景 → E6 增 Y′ 来源要求（真实业务依据或基准化协议）
2. ⚠️ 种子噪声鲁棒性零验证设计——规则引擎粗标带噪，迭代放大错误 → 消融表增噪声注入行 {10,20,30%}
3. ⚠️ 类别长尾分布无应对表述 → method §3.3.2 补 frequency-aware margin 决策 + 敏感性扫描

### 🆕新人审稿人（"6 个月后我要零上下文复现"）
4. ⚠️ related-work.md 元数据自相矛盾：头部 v0.1 / 实际 v0.2；词数 900/实际 1074；C-N 计数 6/实际 9 → 全部校正
5. NOTE fig1 落点犹豫未决 → 定死 Intro 尾部
6. NOTE P0.1 数字在 method.md 与 experiment-skeleton 双维护点会漂移 → method.md Evidence 段加"以 E1 为唯一维护点"指针

### 🕵️主张审计员（"专抓过度主张与引用完整性"）
7. ⚠️ 正文出现内部文件名 [NOVELTY_CHECK_YAOQING_JIA.md]——投稿时内部文档名不得进正文 → 措辞改 "(systematic internal survey; to be migrated to supplementary material)" + 待办登记迁移项
8. ⚠️ outline §0 用结果性动词 "require" 断言未证结论 → 降级为能力性主张 "is designed so that ... can be absorbed"，并写入措辞纪律注记
9. NOTE C6 "≤20%" 表述需按 E4 实测收紧 → 已入待办

### 升级规则应用
无跨人格重复发现（本轮三视角各自独立命中不同盲区），无需升级。

### 验证插曲（诚实记录）
- 修正 C-N 计数时初测 rg 得 6、复核实为 **9 条**：3 条以内嵌形式 `; CITATION-NEEDED:` 标记，正则只命中带独立方括号者。口径已统一为「9 条待补题录」。
- 该插曲反向暴露一个真问题：outline 引用计划与 RW 状态不同步（Mamba-MSQNet 状态冲突 + 跨视角/TIP/GRA 漏列）——已修复同步。

### Round 2 残余风险（写作不可解，移交）
- R1/R2（行为级证据、标题证据）→ 等 P0.2-P0.5
- R4 NTU 验证行 → 用户决策中
- R5 数据集题录 → 文献终审窗口
- R10 Y′ 具体选型 → P0.5 开工前与用户对齐业务依据

**修复提交**: （本轮 commit 号见 git log）

---

## 投稿就绪门（Submission Readiness Gate，v0.3 定义）

以下全部 ✅ 才允许进入投稿流程：
- [ ] P0.2-P0.5 数字归档且回填四件套（R1 解除）——🟡 数字已全部归档回填（E1-E6/两臂/五臂），门禁待用户确认"行为级证据充分性"后勾选
- [ ] E6 结论支撑标题副句，或用户批准标题降级（R2 解除）——🟡 证据已闭合（full 6.07×），正式改判留用户
- [x] 题录补齐：W17 终审 11 条完全解决 + 3 条缩窄为作者列表级，三数据集题录齐备（**R5 解除**；残余作者列表项转入 Scholar 清单 16 条，见 `scholar-checklist-2026-09-03.md`）
- [ ] Scholar/arXiv 人工终审完成并回写核验文档（W1 边界声明解除）——⏳ 清单已导出待人工执行
- [x] 内部引用迁移至 Supplementary Material 完成 ✅（2026-09-03 dffcf95，"internal survey"两处迁移）
- [ ] Data Availability + Ethics + 许可终审声明就位（R7 解除）——🟡 声明正文已装配（dffcf95），许可终审 A-D 项留用户
- [x] NTU 复现行完成或用户书面豁免（R4 解除）✅ **2026-09-02 三流融合 77.97% ≥ 77.18% 预注册线 PASS**（joint 74.30 / bone 71.51 / motion 67.84；证据 `reports/ntu-phaseB-3s-ensemble.json`）

> 状态小结（2026-09-03）：7 门中 3 ✅ + 4 🟡（全部卡在用户终审项/人工 Scholar 执行，零技术阻塞）。

---

## Round 1 + Round 2 评审记录（2026-09-03，装配后成稿评审）

> 完整发现清单与修复留痕: `review-loop-r1-2026-09-03.md`（R1 五视角 8 发现 = 3 MAJOR + 4 MINOR + 1 判定留痕）

### Round 1（结构/主张/引用层）
- 修复: full-budget 措辞消歧 / tab2 正文引用句 / corroborates 措辞精确化（4e30635）
- 判定留痕: AL 数字三处出现各司其职 / 摘要不加 NTU 句 / DA 与 S4.4 措辞张力复核零变更

### Round 2（文风层扫描）
- **AI 腔特征词扫描: 零命中**（delve/moreover/pivotal/crucial/leverage/paradigm shift 等 20 词表全节扫描）
- **被动语态密度: 六节 16%-29% 全绿**（阈值 35%）
- **真长句: A 级 8 处已拆**（04 三 tier 句 83w / E4 61w / 五臂 52w / NTU 协议 44w；05 梯度 83w / AL 50w；01 预览 53w；02 AimCLR 61w；03 三域移位 78w）——数字与措辞零改动，仅句法拆分（0043afd）
- B 级（Limitations L5-L9 的 80w+ 句）保留：信息密度系 galaxy 规范允许，且每条句内含分层论证（主端点/机制/次级信号/边界），拆分会破坏论证链
- C 级伪命中（表格行/Algorithm 块/caption）免修
- 编译: 21 页零错误（0043afd 后复验）

### 终审项移交（全部为用户裁决项，零技术阻塞）
1. GenAI 披露颗粒度（工具名列举 vs 概括——现用概括表述）
2. 许可边界 A/B/C（派生骨架再分发 / NTU 许可留档 / 再分发承诺边界）
3. 伦理原句 D（InterPet4D 数据集文档人体被试伦理句摘录）
4. Funding 标准句（如有资助替换）
5. 作者名单（现为 USER 占位）+ Scholar 清单 16 条作者补全

---

## W17 文献终审记录（2026-08-24）

> 工具链合规：全程 GitHub MCP / arXiv API / 本地文献池与本地数据目录溯源，零 WebSearch（AGENTS.md 硬规则 1）✓

### 完全解决（11 条，均有官方渠道背书）

| 条目 | 题录要点 | 溯源渠道 |
|------|---------|---------|
| InterPet4D | Peng/Song/Liao/Kitani/Koike/Wu, @dataset 2026, HF ohicarip/interpet4d, CC BY-NC 4.0 | 本地数据集目录自带官方 BibTeX |
| Animal Kingdom | Ng et al., CVPR 2022 pp.19023-19034, arXiv 2204.08129 | 官方仓 SUTDCV/Animal-Kingdom README BibTeX |
| APTv2 | Yang/Deng/Xu/Zhang, arXiv 2312.15612；谱系源头 APT-36K NeurIPS 2022 | 官方仓 ViTAE-Transformer/APTv2 README BibTeX |
| SMQ | Gökay/Spurio/Bach/Gall, ICCV 2025 pp.12101-12111, arXiv 2508.04513 | external/SMQ 官方仓自带 BibTeX |
| TCL | Singh et al., CVPR 2021, arXiv 2102.02751（项目页 cvir.github.io/TCL）；官方无代码与 reports/p04 结论一致 | arXiv API（comment 字段 "Accepted in CVPR 2021"） |
| DINO | Caron et al., ICCV 2021, arXiv 2104.14294 | 官方仓 facebookresearch/dino BibTeX |
| PoseC3D | Duan et al., CVPR 2022 Oral, arXiv 2104.13586 | arXiv API id_list 精确查询 |
| 层次化犬行为 | Miyai/Kubo/Saito/Ohno/Kikusui/Nagasawa/Ikeda，NeurIPS **2025 Workshop AI for Animal Communication**，OpenReview EDeOoWN4PT——⚠️ 是 Workshop 论文非主会，正文措辞已按 workshop 书写 | OpenReview 缓存 JSON（Yeping-Hu/ai-workshop-tracker） |
| CrosSCLR | "3D Human Action Representation Learning via Cross-View Consistency Pursuit", CVPR 2021 | 官方仓 LinguoLi/CrosSCLR（72★） |
| Mamba-MSQNet | Fazzari/Romano/Falchi/Stefanini, Ecological Informatics 2024 art.102955；池内出处标注经核验正确；另发现同组 MetroAgriFor 2024 会议版线索 | 官方仓 edofazza/mamba-msqnet README 指定引用 |
| Science 特稿 | Grimm, D., Science news feature（池内已有完整信息，正文标记直接替换） | RESEARCH_LITERATURE.md §1.2 |

### 缩窄保留（3 条，题名/期刊/链接双源锁定，仅缺作者列表+DOI）

YOLO-PetX（IEEE CEECT 2025，全题名经 GitHub 全局代码检索锁定）、Momentum Contrastive Teacher for Semi-Supervised Skeleton Action Recognition（IEEE TIP，Xplore 10820022）、GRA: Graph Representation Alignment for Semi-Supervised Action Recognition（IEEE TNNLS，Xplore 10398229，awesome 目录 TNNLS 分区）。三者均为无公开代码的期刊/会议论文，GitHub 工具链结构性盲区 → 归入投稿前 Scholar 人工终审。

### 终审中的诚实修正

1. ❌→✅ 排除 TCLR 假阳性：arXiv 2101.07974（Dave et al., CVIU 2022）数字 82%/69.9% 与池内「82.7%@10%」不符、venue 不符——不能冒名顶替，最终经标题精确检索锁定真身 arXiv 2102.02751。
2. ⚠️ AK 会议年份澄清：官方 README 证实为 CVPR **2022**（规划期口误记忆为 2024 的风险已消除）。
3. ⚠️ 层次化犬行为论文降级表述：主会 → Workshop（OpenReview venue_id 为准）。

---

## W21 论文实验章诚实修正轮记录（2026-08-24）

> 任务书: `dev-docs/handovers/W21-paper-experiments-honest-backfill.md`（唯一任务书）｜执行: 歆歆（W21 窗口，纯文档零 GPU）
> 性质: 非评审轮，为**诚实性修复轮**登记——清除过期占位污染 + 真证据回填 + 矩阵/Limitations 同步

### 修复内容与溯源

| 项 | 修复 | 事实来源 |
|----|------|---------|
| E5 过期占位污染 | 删除冒充残留 `0.691 ± 0.013`；负结果如实入册（合成层短预算熵未占优 b=100 随机 +7.9pp / b=200 +7.1pp，3/3 seeds 同向；真实池 softmax 全饱和 margin 100.9 vs 合成域 ≈10.8） | `reports/w14-p05-al-efficiency-2026-08-24.md` §3 |
| E6 占位污染 | 回填 C1 真证据：墙钟 7.32×（保守 ≥3×）、精度 +2.27pp、三 seed 全向一致、标注单元打平——合成层 small 档，full 档待 GPU | `reports/c1-decouple-cost-2026-08-24.md` §2 |
| tab3 消融表 | 六行中四行逐格溯源映射（噪声注入/锚点引导/伪标签迭代/主动学习），两行 PENDING 不编造 | P0.3 §4+§4.2+§4.3 / P0.4 §3+§4 / W14 §3.1 |
| Claims-Evidence | C1 ⏳→🟡；C7 保持 ⏳ + 负结果注记 [PENDING-用户措辞裁决]；R2 🔴→🟡；新增 R11（AL 负结果对管线完整性叙事影响，🟠 MAJOR 待用户裁决）；fig4/tab3 状态行同步 | outline.md v0.6 |
| Limitations | 新增 L5（不确定性采样冷启动无优势 + softmax 跨域饱和诊断）/ L6（AK 公开真实层结构约束 4/12 类 + PE ≈4.6 帧/视频，自提取缓解待 Q3 接力）；rebuttal 扩至 6 条 | conclusion-limitations.md v0.2 |

### 本轮裁决移交（用户待决）

- **C7 措辞 [PENDING-用户措辞裁决]**：①降级"探索性发现"写入 §5 分析节；②移出贡献列表。W21 未定稿任何 C7 表述。
- 关联待办联动：C7 裁决后 RESULT-7 语义、Introduction 贡献 bullet 4、rebuttal 预案需一并收口。
- 并行事实登记（本窗执行期间）：协调会话落档用户双裁决（commit `9511d3a`）——A2 warm-start 建册 W23（E5 正证据候选来源，禁触 docs/paper）、B-full worktree 隔离建册 W24；**C7 论文措辞裁决不受 A2 影响，仍独立待用户**。

### 裁决落地追加（2026-08-25 · 同窗）

用户对 C7 措辞拍板**选项①**（歆歆推荐被采纳）：降级"探索性发现"写入 §5 分析节。

| 文件 | 变更 |
|------|------|
| experiment-skeleton.md v0.6 | E5 标题/目的句/处置块改裁决记录；效率主张正文与摘要禁用；升级通道注明（W23 warm-start 或 full-budget 正证据 + 用户再裁）；tab3 AL 行标 C7↓；fig4 落点迁 §5 |
| outline.md v0.7 | 矩阵 C7 行标 🔻 降级（落点 →§5）；§3 故事线 #4 与 §4.1 Intro 贡献 bullet 4 计划缩窄（主动学习转探索性发现）；摘要 [RESULT-3] 原"标注预算节省比"失据标记待重选源（候选：SMQ IoU 1.53× / 伪标签 +17.9pp / C1 墙钟 ≥3×）；R11 ✅ 缓解落地 |
| conclusion-limitations.md v0.3 | 结论模板删除"100–200 片段预算内达标"句式（RESULT-7 改指语义层精度）；L5 rebuttal 更新为裁决①表述 |

> Introduction.md 本体属 W21 禁触领地（等 Q3 数字统一终稿窗口处理），bullet 4 与摘要 [RESULT-3] 的实际改写已在 outline 计划层登记，由终稿窗口执行。
> 升级条款：W23 或冷启动 full-budget 若产出 E5 正证据，须经用户再裁方可恢复 C7 主张地位——禁止窗口自行升级。

### 完成标准自查（任务书 §5）

- [x] 全文检索无 `0.691 ± 0.013` 冒充 E5/E6 残留（E4 行为合法数字保留）
- [x] 每个 E5/E6/tab3 数字可溯源到具体 reports 文件+字段
- [x] C7 措辞裁决项显式标记待用户
- [x] 三层口径零混排（E5/E6/tab3 各格层级标注齐备）

**提交链**: `64a1c8a`（E5/E6）→ `7d69b15`（tab3）→ `afd7b0c`（矩阵+Limitations）→ 本条登记。
---

## R8 对抗审稿轮（2026-09-04，增量轰炸本轮新增内容；双通道：程序化数字对账 + 独立恶意审稿 agent + 引用权威实查 agent）

**通道 1 程序对账**（35 项 tex 数字 vs reports JSON）：29 过；真错 1 处（E9 +8.0→**+8.1pp**，74.11−66.05=8.06）；其余 5 项为格式差非错误。
**通道 2 恶意审稿**（Reviewer-2 人格，只打新增段落）：判定 **Major Revision**——2 CRITICAL + 9 MAJOR + 6 MINOR，全部有效并修复：
- **CRITICAL-1 类别空间混淆**：EP3 原始 +3.57pp < chance 位移 +4.17pp（12→8 类）→ 补 8 类同空间对照（v1 full 重算 25.96% → 同空间差 **+11.54pp**，above-chance 13.5→25.0）；归因修复后于 top-1 成立，macro-F1 反向双报；协议 §7 dated 修订。
- **CRITICAL-2 泄漏披露违约**：协议承诺入文的 Y_CKPT 源视频披露正文缺失 + "independent" 矛盾 → E7b 补披露句，independent→pre-registered parallel。
- **CRITICAL-3（引用实查 agent，最重）TCL 假引用**：82.7%/88.6%/NTU60 经 arXiv 2102.02751 原文 15 页 PDF 全文检索**证伪**（NTU 0 命中/82.7 0/88.6 0；真实数据集=Mini-Something-V2/Jester/Kinetics-400/Charades-Ego）；bib 作者亦错（正确=Ankit Singh…Abir Das，前轮"Jathushan/Amlan"系张冠李戴）。该数字系 outline 池内继承、R7 只核作者未核数字、待办挂账未执行——**教训：作者核验≠数字核验，"投稿前复核"待办必须闭环不得挂账**。处置：正文 E4/E9 删除、Intro 联动、method/related-work/outline/skeleton/两报告全链清污+勘误附注、协议 §6 修订；E9 判据仅依赖预注册 90% 线（结论不受影响）。
- MAJOR 修复：微调句双指标限定（macro-F1 16.73>7.83 反向如实报，superiority 限双指标一致区）/v2 macro-F1 腰斩+sit n=2 披露/artifact→hypothesis/tab3 full 列 "(deterministic, untested)"/both-arms-beat-scratch 限 spc2/only-backbone→+预处理/30× 删除改 28× 定义/probe ceiling→reference/协议依赖+单子集披露/Intro 机制限定语/Abstract 补绝对锚点。
- MINOR 修复：ten-seeds 修饰语/13% 口径（18/141）/适配披露入文/tab2 行注/微调种子数/geometry 断言降格（无 CKA 实测）。
- 附带：AimCLR++ 77.2→**80.9**（官方仓库证伪，77.2 系 CrosSCLR 串行误引）；79.18 归属措辞；aimclrpp 题名按 CrossRef；NTU60 补引 Shahroudy2016。
**查无问题维度（明示）**：significant 措辞零滥用/13% vs 6% 无混引/K9 预注册句忠实/E7 n=10 数字逐位吻合。
**修复后状态**：21 处正文编辑+3 处 bib+8 处 truth 源+2 协议修订+2 报告勘误；重编译验证；证据-主张对齐恢复。

---

## R9-R11 视觉修复轮（2026-09-05，绘图技能加载：diagram-design + scientific-visualization + anti-defensive-writing）

- **触发**：用户令真实加载绘图 skills 优化图。judge 首轮 4/4 FAIL → 根因=大画布缩放致落地字号 2.0-2.6pt（Elsevier 要求 ≥6pt）→ 四图全部按印刷尺寸重设计（figsize 3.4in 1:1）。
- **R9**：修局部碰撞（fig1 框宽/换行/横幅缩短/标题出线、fig2 hub 缩字+辐条标签外移、fig3 标签门限+agg 下移、fig4 注释移位+字号 6.2）→ judge 复验 fig4 PASS，余 3 局部 fail。
- **R10**：fig1 输入标签缩短、fig2 辐条标签贴 hub 外沿+站盒 23、fig3 agg 移左上+caption 删 band-text 承诺。**过程事故**：补丁脚本末段路径错抛异常 → `&&` 链断 → 图未重生成即送 judge → judge 报"修复未落地"（judge 正确）。教训：**复验前必须验证工件时间戳/内容确实更新**（pymupdf 提取 PDF 文本核对），不能假设脚本跑了。
- **R11**：重生成+重编译+pymupdf 核对嵌入内容后送 judge → **4/4 PASS，视觉门通过**。
- **anti-defensive-writing 对新增段落复查**：E7b/E9 新披露句逐句归类=第 3/5 类（真实方法学限定/基于证据的限定），按 Preserve Necessary Precision 铁律全部保留；无第 1/6 类（不必要免责/冗余澄清）命中——本轮修复本身是"加必要披露"，非防御性对冲。
- 引用数字清扫（R8 续）：BCST-GCN 94.43→95.36（张冠李戴证伪）、Grimm USD 12,000→tens of thousands（原文无此数）、AK 329/APTv2 83,304/≈4.6 帧三处归属措辞改"本仓实测"口径、grimm bib 补 DOI/volume/pages、NTU60 补引 Shahroudy2016、AimCLR++ 77.2→80.9。

---

## R12 全文三维度对抗审稿轮（2026-09-05，工具箱技能 ars-adversarial-reviewer + scholar-critique-manuscript；三独立 agent 并行：新颖性/统计方法/可复现工件）

**新颖性线（4 MAJOR + 3 MINOR，全部修复）**：
- 首次性主张被 arXiv 2603.06201（Point-Supervised Skeleton Action Segmentation，原型伪标签+聚类，时序骨架）实质逼近——**本会话 arXiv API 独立复核 6 篇新引全部真实**（含 ID→标题精确映射，防 agent 幻觉引入假引用）。修复：§2.3 增最近邻区分句（"claim firstness for that combination, not for prototype pseudo-labels per se"）、Intro bullet 2 加 to-our-knowledge + 不可排除未发表工作、highlights.tex 恢复 hedge（此前与 main.tex 版内不同步=独立文件无限定断言）。
- "Supplementary Material" 三次引用但附录被注释不存在 → **启用双附录**：app:novelty（14+1 候选英文矩阵表）+ app:repro（真实脚本名复现链，逐一核对存在性）。
- §2.1 绝对 gap 句被 2509.12193（灵长域自适应预训练）/2507.09513/2207.10553 击穿 → 软化+补引；§2.2 骨架 MAE/foundation model 线（2209.02399/2508.12586）整段缺席 → 补入并正交性区分（冻结的是物理编码器，可修订的是语义层，度量是迁移成本非表征质量）。
- Intro "cold-start near chance at 20 clips" 无落点 → 补 w23 实测 ~7.8% 对照入句；"≈0.30"→0.32（压低 null 放大表观增益，违反自家数字纪律）。

**统计线（1 CRITICAL + 3 MAJOR + 1 MINOR）**：
- **CRITICAL：§4.2 声称 Holm-Bonferroni 校正但从未执行**——修复=真做：`scripts/run_r12_artifacts.py` 产出 `reports/r12-holm-eightclass-2026-09-05.json`，p10 族（m=6）校正后 spc2 top1=0.012/spc4 top1=0.047（仍显著）/mf1 parity=0.77，p12 族全 0.012；正文改印校正 p；"fold-level"→"seed-level"。
- 25.96% 无工件+措辞失实（"re-scoring"实为 per-class 重组）→ 工件入仓 + 正文改 "reweighting...on v1's validation clips" + 披露残余 clip-set 混淆。
- E9 置信门接受 ~95% 池——"confidence-filtered"近恒等 → 正文披露 + **新增 L11**（Ten→Eleven limitations）；40,091 vs 官方 40,128 差异披露。
- "statistically equivalent"（E5/结论）→ "inside the pre-registered noise band"（非 TOST 不借术语）。

**可复现线（4 MAJOR + 2 MINOR）**：
- **77.97% 证据 JSON 从未 commit**（仅存 W33 worktree）→ 已拷入 `reports/ntu-phaseB-3s-ensemble.json`（top1=0.7797/top5=0.9578/n=16487/三流 74.3/71.51/67.84 与正文逐位一致）。
- "committed before the build/experiment" 被 git 时间戳证伪（协议与结果同批 commit）→ 措辞降级 "frozen before the build (protocol and results landed in the same repository batch)"——预注册实质（构建前冻结）保留，可验证性声称诚实化。
- "one-command chains" 对审稿人不成立（硬编码 k9 私有仓路径+权重不入库）→ DA 措辞改"released scripts + reproduction appendix + 外部数据按许可自取"；README 增 Reproduction 节；app:repro 列全链真实脚本名。
- 1.62×→1.61×（3 处，选择性进位）；96.6% 补 "single run; ±0.5pp observed"。

**验证**：20 项程序化断言全过（含 77.97 可追/工件存在/措辞残留清零）；编译 29 页零错误零 undefined（附录 +2 页，仍在 PR 20-35 页窗口内）。
**判定**：R12 前=Major Revision 级（Holm 虚假声称+无工件数字+证据缺失为三大硬伤）→ 修复后证据-主张对齐恢复，且新增 6 条真实引用强化文献定位。

---

## R13 恶意评审轮 + R14 图终验轮（2026-09-05，ars-adversarial-reviewer 三人格机制 + scholar-critique-manuscript/figures 框架）

**R13a 恶意全文（Reviewer-2 人格，回归攻击 R12 修复+全文遗留）**：2 CRITICAL + 19 MAJOR + 12 MINOR，逐条独立验证后全部处置：
- CRITICAL-1 §2.2 新句自带 "Supplementary Material" 死指针（R12 修 §2.3 时漏改 §2.2）→ 改 Appendix ref。
- CRITICAL-2 E1 "±4.45%" 与存档 JSON 重算不符（fold std ddof=1=4.04）→ 改 ±4.04 并注明口径。number-index 的 quickref 值系历史错误——**头条不确定度也须对账**。
- MAJOR 全修：novelty 计数三方互斥（14/18/19→统一 18+1=19 行口径）；复现链幽灵 E8/合成层整层缺链/NTU 错序→重写；L3 与已完成 re-verification 矛盾→改写；"full taxonomy coverage"摘要/结论升格→toward；"any recognition task"→限定两域已验证+设计主张；E7 "confirms"→"supports on top-1"；摘要 kNN 补 subject-identification probe 标注；12-class→12-class protocol 9-with-samples；Algorithm 停止条件补 precision-drop、τ 注明自适应；fig2 队列=部署扩展+B 定义入 caption；44.90 补单 seed+±4.81；0.691 补 r1 操作点+停止规则；purity 口径纠错（InterPet4D 7 标签空间非 12-class）+ majority 基线 0.4858 主动披露；pool accuracy→precision 统一；"below both arms' std"→within one arm's；fig4 种子级 "3/3" 证伪（b=100 seed43 entropy 赢）→2/3+caption "no detectable advantage"；warm-start 82% 先验来源（全监督同分布模型）披露；C1-C7 标签映射句入 §4.1；三图正文引用补齐（fig:pseudoloop/fig:budget/fig:al-efficiency 原零引用=期刊硬伤）。
- 干净维度如实记录：Holm p 逐位一致、62 处 \ref 零悬空、34 cite 全在 bib、无 "we will" 残留、"Eleven limitations" 计数正确。

**R13b 图内容批判（scholar-critique-figures 框架）**：1C+10M——fig4 种子断言（已并入 R13a 修）；fig2 辐条语义错误（全 6 站画写回辐条，但 Assign/AL queue 不写状态）→ 只保留 Ω/P/A 三辐条、κ 路由标签移环弧、focal 移至 pool 站；fig1 缺 warm-start 元素+接口箭头落错盒（指 self-training 应为 clustering）→ 双修；fig3 正文 zoom 从句失效→删；**新增 Figure 4 budget-retention 散点**（论文核心低资源主张首次获得单图视觉证据，5 点 4 层直读 JSON）；fig4-AL 增面板 (b) warm-start 臂（更强负证据入图）。

**R14 视觉终验（judge 三轮）**：第一轮 4/5 fail（fig1 弧钩残留/fig2 hub 裁切+站名溢出/fig4 y 轴 `\%`+图例碰撞/fig5 注释穿线+caption 溢出）→ 修复中踩坑：**副题删除正则误吞 PDF savefig 行**（PNG 更新 PDF 未写，judge 正确报"未落地"——复验再次证明工件时间戳/内容核验不可省）；第二轮 fig4 图例右上仍撞（窄图内无空位）→ 移轴外下方两列；第三轮 **5/5 全过，视觉门通过**。页脚 -1 偏移为 cas-sc 首页不编号既有行为，非缺陷。

**编译**：31 页零错误零未定义引用。

---

## R15 全文逐行审计轮（2026-09-05，四 agent 并行逐行覆盖全部 tex + 实现真源交叉核对）

**性质**：首轮真正的逐行地毯式审计（此前均为增量/主题式）。共 60+ 发现，全部验证后处置，最重一批：
- **03-method 实现-描述失配（5 高）**：①伪代码/正文写"最近原型分配"，主线实现是**分类头 Ω 分配+原型路共识门**（tcl_selftrain.py L321）→ 改写双路结构；②"round-adaptive τ"与实现相反（τ* 首轮冻结、类级频率缩放）→ 改"fixed once...scaled per class"；③池"add"实为 REPLACE 重过滤 → 改"re-filter (replacement)"；④precision-drop 停止消费**共识伪 GT 精度**且 patience=2 → 伪代码与 §3.3.3 如实披露（并挂实现侧防火墙冲突为已知问题）；⑤"temporal contrastive self-training in the spirit of TCL"——实现无时序对比（tcl_head 纯 CE，TCL 对比为 stretch goal）→ 降级为"iterative retraining...tradition motivated by TCL (step not used here)"；⑥warm-start 三修饰语（this stage/previous taxonomy/same distribution）均为 R13 我加的未经背书润色，与 ADR 0005（noise-offset）矛盾 → 回真源+精确化。
- **数字/归属（R15a/c/d）**：intro 7.8% 直接对比违反 W23 comparability 护栏→回定性措辞；79.18% 归属错（系官方仓复测非 journal-extension）；"19 table rows"实为 19 候选 8 行；"never below 4.00×"严格为假（最小 3.9993）→3.99×；"official split lists 40,128"无一手源→删具体数；"restrict all public-real claims"与 E7/E7b 自相矛盾→限定该聚合分数；"≤1pp at operating K"在 K=14 实为 −2.4pp→如实分列；L5"all three seeds"（06 漏改）→2/3；33,099"disclosed in inventory report"虚假指针→disclosed here；不平衡开关"is evaluated"虚构完成态→registered (not yet run)；L7 train 支持数错标→full-list/train 分列；L9 自矛盾→legitimate gain；L2 primates 与 NTU 冲突→限定 real animal data；"decays monotonically"与 +0.15 回升矛盾→to statistical equivalence；variance→std 3.8×；tab2 caption"every public-real aggregate"过度→four-class；E6/E8 编号缺口加 ledger 说明句；C6/C7 映射对齐 ledger；ddof 约定入 §4.2；TP-CanineNet 年份 CrossRef 实证 2026+题名回正。
- **hedge 补齐**：3 处裸"never/No existing"加 to our knowledge（与 L56 自洽）；"is bounded"→mitigated；"grows a complete taxonomy"→labeled coverage；"overviews"→illustrates；caliber/tier 混用修正；摘要压缩至 ~205 词并删冗余；nocite 脚手架删除；highlights 两副本同步 synthetic-offset。
- **验证**：31 项残留断言清零（1 项误报为其他条目合法 2025 年份）；编译 0 错误 0 未定义引用；本轮无图改动（视觉门维持 R14 的 5/5）。
- **教训**：逐行审计与主题审计发现面几乎不重叠——R15 抓出的 03-method 五条高危全是前四轮"增量打击"盲区（方法节自 R7 后未被逐行攻击过）。

---

## R16 端到端协议诚信修正轮（2026-09-05，ars-adversarial-reviewer 三人格 + 实现真源三角核对；本轮为学术诚信级）

**触发**：Saboteur 人格攻击"论文描述的方法 vs 实际跑的方法"，命中两发 CRITICAL（逐行读码亲验，非 agent 单源采信）：
- **CRITICAL-1 端到端头消费真标签**：p07/p08/p10/p12/p14 的最终分类器在**池片段真标签**上重训（run_p07 L137-139 `train_linear_head(emb, labels, anchor|pool)`），伪标签仅做选择——"94%@13%/99.5%@10%/88.6%@6%"头条主张与实现不符（实际消费 60-99%/96% 训练标签；warm-spc2 seed43 池 122/123→140/141 真标签，top1=0.3393 与全监督臂逐位同——铁证）。
- **CRITICAL-2 停止规则 oracle + 披露反写**：`_record` 的池精度（消费 truth_all=真标签）驱动 precision-drop 停止；E9 正文却披露"NTU 无真值、停止用 head-estimated precision"——双重失实。
**处置（修复轮自身按铁律逐句回验）**：
- 实现侧：`tcl_selftrain.py` 增 `precision_stop`（False=oracle 精度仅诊断不入控制路径；默认 True 保 P0.4 共识语义）与 `head_calib`（GT 无关锚点侧逐轮再校准，诊断用）。
- 新驱动 `run_r16_endtoend_pseudo.py`/`run_r16_ntu_pseudo.py`：最终头=种子真标签∪池伪标签；停止=收敛/预算；纯监督参照不动。重跑 v1/v2×10seeds + NTU×3seeds（工件 `r16-endtoend-pseudo-2026-09-05.json`/`r16-ntu-pseudo-2026-09-05.json`）。
- **修正后结果**：AK v1 warm spc2 **9.8%±7.5**（chance 11.1%）/aimclr 15.2/scratch 8.4——端到端低资源主张在犬科层**撤回为负边界**；v2 同（13.1/17.6/13.8 vs chance 12.5）；head_calib 诊断不救（池精度 9-12%）→失败归因伪标签质量非门控尺度。**NTU 修正后 67.5%±0.15=保留率 90.6%≥预注册 90% 线——GENERALIZES 判据在更严诚实协议下存活**（增益 +8.1→+1.4pp）。
- 论文联动：摘要/Intro/E7/E7b/E9/tab2/tab3 行2+段落/L1/L9/L11/fig5（重绘 tier-dependent 叙事）全装配；E4 补自家 Holm（校正后 p=0.090，工件 `r16-holm-p04`）；EP3 披露求解器路径噪声（标签重编码 37.50→35.42，归因锚定同空间对照）。
- 协议文档：**只追加 dated 修订**（NTU §7、AKv2 §8），零静默改；p08/p10/p12/p14/w14 报告追加勘误节。
**同轮其余发现（三人格+New-Hire+引用实查 agent 并行，逐条验证后处置）**：
- MAJOR：E 编号与 owner 台账冲突（tex E5=transition vs skeleton E6；"released pre-registration ledger" 假指针→改真实台账措辞+E5/E8 去向注）；L10 "decaying monotonically" R15 修复未传播残留→改；tab3 锚点行 "≤1pp at operating K" 与 E3 K=14 −2.4pp 矛盾→如实分列（并纠 R15 引入的 "main-run K=14" 错标——主跑是 class_mean）；03 "is ablated" vs 05 "not yet run" 自相矛盾→registered；共识门 AK/NTU 恒惰性（STANDING_LABEL 不在类空间）→方法+实验双披露；五臂段 frozen 臂错标 §4.3→改 full-budget head-retrain；伦理句 "canine pose corpora cited in §4.1" 假指针→改写；复现链缺 run_p05_public_real_full12_endtoend.py→补；附录段 5 处 overfull hbox→\sloppy 清零；fig5 脚本 syn_full 硬编码回退恒触发→真读 summary 键；skeleton E1 ±4.45/AL 3/3 两处 stale→同步（w14 勘误：b=100 实为 2/3，seed43 熵 0.7909>随机 0.7818）。
- MINOR：main.tex `\end{document}` 后重复 bib 块（编辑事故）删；novelty 段重复句删；TIP 年份表内删（bib 2024 经 DOI 实证，日志 2025 为 outlier）；"; A motion-word"/"a five new arms"/"and left to future work" 语法；"legal views"→valid；"untested"→not significance-tested；top1--top2→top1$-$top2 统一；"three data calibers"→两 exercised tiers+human benchmark；"confirms matches official reference"→meets pre-registered criterion；96.6% ±0.5pp 无源→删（single run, seed 42）；"released-checkpoint table"→official repo linear-eval table（AimCLR-v2 README 实证）；BCST-GCN "self-collected"→"compiled from online public videos"（原文自述网络公开平台采集）；DA 补 \ref{app:repro}；main.tex 注释 stale（六行/作者待补）更新。
- **外部数字原文 PDF 核验（引用实查 agent，pymupdf 全文检索）**：ASBAR 75.3% ✅（eLife PDF 摘要命中，PoseConv3D/PanAf500 大猿）；BCST-GCN 95.36% ✅（Frontiers PDF Table 9）；AimCLR 78.9% ✅（arXiv 2112.03590 Table 2 3s-linear-eval）；79.18% ✅（官方 README Trained models）；80.9% ✅（AimCLR-v2 README Linear Eval 表——provenance 措辞已修）；APTv2 2,749/41,235 ✅（arXiv PDF 摘要逐字）；33,099 ✅（=Animal Kingdom 论文表格，tex 归属句已明确）；Grimm 特稿 ✅（tens of thousands/>half/服务犬主体三子检查全过，"service or assistance dog" 忠实）；InterPet4D 伦理三句 ✅ 逐字（HF raw README L167-169）；残留扫描 82.7/94.43/77.2/12,000/40,128 零命中 ✅。
**验证**：断言脚本 `scripts/r16_assertions.py` 全过（48 残留清零+15 新数字在位+refs/cites 结构）；编译 32 页 0 错误 0 未定义 0 overfull；fig5 工件核验 mtime+pymupdf 刻度抽取。
**判定**：R16 前论文处于**拒稿级隐患**（头条数字与实现不符+披露失实）；修正后证据-主张对齐恢复，代价=低资源端到端主张收缩为"跨域 NTU 90.6%（预注册线存活）+犬科层负边界"——诚实叙事与 L9 机制诊断闭环。
**教训（第 N+1 条）**：①组件级对账全绿≠管线级协议诚实——"最终头用什么标签"这类**数据流问题**只有读驱动代码才能暴露，数字对账与主题审计都覆盖不到；②oracle 停止规则会**掩盖**而非修复迭代退化（p07 的 precision_drop 恰好在坍缩前刹车，使真标签头看起来正常）；③修复轮文本（R15 的 "main-run K=14"、"released pre-registration ledger"）本身继续产幻觉——每句修复文本落盘前对真源回验纪律不豁免。
