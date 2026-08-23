# W5 交接文档 — P0.6 论文初稿启动（与 P0.2 并行 · 挂接学术工具集）

> **你是 W5 窗口**。读完本文档即开工。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md` v1.2 → `dev-docs/project-brief.md` → `research/PAPER_POSITIONING.md`。
> **本任务核心变化**：论文写作全程以 `D:\Desktop\数模竞赛` 学术工具集为方法论参照系（详见 §3）。

---

## 1. 任务目标（一句话）

搭建 Pattern Recognition 英文初稿骨架，完成**材料已齐备的章节**：Related Work（吃 W1 成果）+ Method 框架（吃双仓架构口径）+ Experiment 骨架（数字留占位符）——**写作质量与结构遵循学术工具集的技能规范**。

**明确不做**：实验数字填充（等 P0.2-P0.5）、Introduction 终稿、投稿格式排版。

## 2. 执行链（每件产出 = 先读对口技能 → 按其规范写）

| 步骤 | 产出 | 先读的工具集技能（绝对路径） |
|------|------|---------------------------|
| 0 | 工具集方法论速览 | `D:\Desktop\数模竞赛\数学建模全流程套件\skills\galaxy-ml-paper-writing\SKILL.md`（43KB，ML 期刊论文主方法论）+ `D:\Desktop\数模竞赛\参考论文使用指南.md`（62 篇优秀论文统计规律） |
| 1 | `docs/paper/outline.md` 全文大纲 | `skills\paper-plan\SKILL.md` + galaxy-ml-paper-writing 的章节组织节 |
| 2 | `docs/paper/related-work.md` 英文初稿 | `skills\sci-literature-review\SKILL.md`（若在）+ galaxy-ml-paper-writing 的 Related Work 节 |
| 3 | `docs/paper/method.md` 英文初稿 | galaxy-ml-paper-writing 的 Method 章规范 |
| 4 | `docs/paper/experiment-skeleton.md` 三层口径骨架 | `skills\analyze-results\SKILL.md`（结果呈现规范） |
| 5 | 自审一轮 | `skills\auto-review-loop\SKILL.md` 的审稿清单 + `skills\check-citations\SKILL.md` 引用核查规则——**人工执行其检查表**，把审出问题修掉再提交 |

> 技能文档若与列名有出入，在同目录找最接近主题者并记录实际所用；全部路径只读。

## 3. 跨仓纪律（硬边界，违者返工）

1. `D:\Desktop\数模竞赛` **绝对只读**：只允许 Read/Grep 其 SKILL.md 与规则文档
2. ❌ 禁止执行其任何脚本/engine/tools（那是独立工作区的能力，如需自动化引擎由用户单独在该目录开会话）
3. ❌ 禁止向其工作区写任何文件
4. 本任务所有 Git 操作只在 psd-framework 仓

> 合规依据：AGENTS.md 跨仓条款「只允许文档指针」——读文档学方法论 = 文档指针式使用 ✓

## 4. 质量门禁（从工具集提炼，标注了适配性裁剪）

来自 `参考论文使用指南.md` 的可编码规则（⚠️ 原为中文数模语境统计规律，用于 PR 英文期刊时按下表裁剪执行）：

| 规则 | 英文期刊适配 |
|------|-------------|
| 摘要四要素齐全（背景/方法/结果/结论）+ ≥3 个数值结果 | ✅ 直接适用（PR 摘要惯例一致）；数值结果暂写占位符计数 |
| 图表白底+浅灰网格、低饱和淡彩、单图 ≤6 色、避红绿组合 | ✅ 直接适用（后续画图阶段生效，先写入 experiment-skeleton 的图表规范节） |
| 短句占比 ~60-65%、被动语态 ≤4%、学术词汇密度 ≥1% | ⚠️ 英文改写为：句子均长 ≤25 词、被动语态慎用、避免 AI 腔（参考工具集 anti-AI 思想但不必跑其检测器） |
| 引用必须有真实出处 | ✅ 铁律：不确定标 `[CITATION-NEEDED]`，配合 check-citations 清单自查 |

## 5. 引用材料地图（本仓，全部只读）

| 材料 | 用途 |
|------|------|
| `research/NOVELTY_CHECK_YAOQING_JIA.md` §3 排查表 | Related Work 三近邻差距：MAC-Learning / Skeleton-to-Image / TP-CanineNet |
| `research/RESEARCH_LITERATURE.md` | 动物行为识别引用池（17 篇带 DOI/URL） |
| `research/PAPER_POSITIONING.md` | 标题/创新点/组件来源表 |
| `reports/p01-aimclr-2026-08-23.md` | P0.1 行真实数字（20.89% vs 8.33%）可直接填入 |
| `docs/DATA_LOCATIONS.md` | Datasets 节描述（实测值：AK 329 视频、APTv2 ~83K） |

## 6. 边界（与 W4/W6 并行互斥）

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `docs/paper/**`（唯一 owner） |
| ❌ 禁触 | 一切代码/配置/`psd/`/`scripts/`/`configs/`/`external/`/`reports/`/`dev-docs/**`（只读除外）/`docs/DATA_LOCATIONS.md`/`D:\Desktop\数模竞赛`（见 §3） |
| 环境 | **零 Python 依赖**——不要动 `.venv`（W4 主用） |

## 7. 完成标准与 Git

- [ ] 四件套齐全，Related Work ≥800 词英文成稿且经 auto-review-loop 清单自审一轮
- [ ] 所有引用条目有真实出处（仓库名/DOI/arXiv 号）
- [ ] 大纲中每节标注「已可写 / 待 P0.x 数据」
- [ ] 提交：`feat: P0.6 论文初稿启动——大纲+Related Work+Method框架+Experiment骨架（学术工具集方法论）`
- [ ] 完成后在回复中请用户评审 outline.md 故事线

## 8. 升级路径

- 工具集某技能文档缺失/不可读 → 换同主题技能并在 CN-NOTES 记录；整体方法论冲突 → 上报用户裁决
- 发现 W1 结论不足以支撑某段 Related Work 表述 → 引用原文并标注疑问，不自行改判
