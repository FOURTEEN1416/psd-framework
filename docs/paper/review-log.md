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
