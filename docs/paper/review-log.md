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
- [ ] 10 处 [CITATION-NEEDED] + 3 个数据集题录全部补齐（R5 解除）
- [ ] Scholar/arXiv 人工终审完成并回写核验文档（W1 边界声明解除）
- [ ] 内部引用迁移至 Supplementary Material 完成
- [ ] Data Availability + Ethics + 许可终审声明就位（R7 解除）
- [ ] NTU 复现行完成或用户书面豁免（R4 解除）
