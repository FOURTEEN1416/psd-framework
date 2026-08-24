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
- [ ] P0.2-P0.5 数字归档且回填四件套（R1 解除）
- [ ] E6 结论支撑标题副句，或用户批准标题降级（R2 解除）
- [x] 题录补齐：W17 终审 11 条完全解决 + 3 条缩窄为作者列表级，三数据集题录齐备（**R5 解除**；残余 3 处作者列表待 Scholar 终审并入下一项）
- [ ] Scholar/arXiv 人工终审完成并回写核验文档（W1 边界声明解除）
- [ ] 内部引用迁移至 Supplementary Material 完成
- [ ] Data Availability + Ethics + 许可终审声明就位（R7 解除）
- [ ] NTU 复现行完成或用户书面豁免（R4 解除）

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

### 完成标准自查（任务书 §5）

- [x] 全文检索无 `0.691 ± 0.013` 冒充 E5/E6 残留（E4 行为合法数字保留）
- [x] 每个 E5/E6/tab3 数字可溯源到具体 reports 文件+字段
- [x] C7 措辞裁决项显式标记待用户
- [x] 三层口径零混排（E5/E6/tab3 各格层级标注齐备）

**提交链**: `64a1c8a`（E5/E6）→ `7d69b15`（tab3）→ `afd7b0c`（矩阵+Limitations）→ 本条登记。
