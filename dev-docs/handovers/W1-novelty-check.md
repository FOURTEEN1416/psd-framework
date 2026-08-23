# W1 交接文档 — 姚青 JIA 创新性核验（GitHub-First）

> **你是 W1 窗口**。读完本文档即开工，无需等待其他窗口。
> 必读顺序：本文档 → `AGENTS.md`（硬规则宪法）→ `dev-docs/HANDOVER.md`（项目全貌，§9 硬规则速查）。
> 本任务为**纯只读调研**：唯一产出是新调研文档，禁止触碰任何其他文件。

---

## 1. 任务目标（一句话）

用 **GitHub 工具链强制调研（WebSearch 零容忍）**，判定核心创新主张是否成立：

> 「图像域小样本伪标签方法（姚青 JIA：VFM 骨干 + 锚点学习 + 自监督聚类 + 伪标签迭代）迁移到**时序骨架域**」是否有人做过？

结果决定 P0.3 设计与论文框架——被占坑越早知道越好。

## 2. 方法本体参照（内部共识，作为比对基准）

来自 `research/RESEARCH_FOUNDATION_SYSTEM_VISION.md` 发现 4：

| 姚青图像域组件 | 时序骨架域对应 |
|---------------|---------------|
| VFM 特征（DINO/MAE 类视觉基础模型） | 骨架自监督编码器（AimCLR / SMQ VQ 嵌入类） |
| 少量锚点（人工标注种子） | 规则引擎高置信样本 + 20-50 人工黄金样本 |
| 自监督聚类伪标签 | 骨架特征空间聚类 + 伪标签迭代 |
| 置信度筛选 | 同左 |

**核验对象是这个组合迁移到时序骨架域，不是单个组件**（单组件各自都有文献，见 RESEARCH_LITERATURE.md）。

## 3. 执行步骤

### 3.1 先读已有综述，避免重复劳动
`research/RESEARCH_LITERATURE.md`（37 仓库调研结论）——已知 TP-CanineNet（2025，视频域伪标签）等条目，你的任务是找**它没覆盖的骨架时序域工作**。

### 3.2 GitHub 检索矩阵（全部走 GitHub MCP 工具，禁 WebSearch）

至少覆盖以下 6 个维度，每个维度 ≥2 组查询词（中英关键词自行扩展）：

1. **直接命中**：`few-shot skeleton action recognition pseudo-label`、`semi-supervised skeleton based action recognition`
2. **VFM→骨架迁移**：`vision foundation model skeleton action`、`DINO skeleton representation`
3. **动物行为骨架域**：`animal behavior skeleton few-shot`、`pet action recognition pose`
4. **awesome 目录发现**：搜 `awesome skeleton based action recognition`，进目录扫 semi-/few-supervised 小节
5. **TCL/AimCLR 生态**：搜其 fork 网络 / citing 仓库中的半监督变体
6. **伪标签时序**：`pseudo label time series clustering anchor`

### 3.3 逐仓验证（防误判，缺此步结论无效）

对每个疑似占坑仓库记录：最近 commit 时间、stars、open issues、README 中对应论文名/链接、方法是否真为「锚点+聚类伪标签+骨架时序」。活跃度停滞 >2 年或方法不符的直接降权。

### 3.4 三态结论（必须落其一）

| 结论 | 含义 |
|------|------|
| ✅ 首次性初步成立 | 检索矩阵全覆盖后未发现组合迁移工作；列出最接近的 3 个工作及差距分析 |
| ❌ 已被占坑 | 给出占坑仓库/论文证据链接 + 与本项目方案的异同表 |
| ⚠️ 证据不足 | 说明盲区，列出下一步扩大检索的具体计划 |

## 4. 边界（并行窗口互斥，严格执行）

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `dev-docs/research/NOVELTY_CHECK_YAOQING_JIA.md`（新建，唯一产出） |
| ❌ 禁触 | `stage-plan.md`、`project-brief.md`、`PAPER_POSITIONING.md`、`docs/`、`psd/`、`scripts/`、`configs/`、`external/`、`dev-docs/handovers/**`、`DATA_LOCATIONS.md` 及一切代码/配置 |

> 若结论影响其他 truth 文档（如 PAPER_POSITIONING 第 12 行「待核验」），在 NOVELTY_CHECK 文末列「建议回写清单」，由用户或收尾窗口统一执行，**不要自己改**。

## 5. 完成标准与 Git

- [ ] ≥12 组有效检索查询（记录查询词与命中数）
- [ ] ≥8 个候选仓库逐一验证并形成证据表（仓库名/活跃度/方法比对/结论）
- [ ] 三态结论明确 + 最接近工作差距分析
- [ ] 提交：`git add dev-docs/research/NOVELTY_CHECK_YAOQING_JIA.md && git commit -m "docs: 姚青JIA创新性核验报告——<三态结论摘要>"`
- [ ] 遇 `index.lock` 冲突等待重试；本仓无远程，**禁 push**

## 6. 卡住升级

同一检索方向连续 3 次无有效命中 → 换维度；整体卡住或发现重大占坑证据 → 立即在报告中标注并在回复中向用户升级，不等收尾。
