# W5 交接文档 — P0.6 论文初稿启动（与 P0.2 并行）

> **你是 W5 窗口**。读完本文档即开工。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md` v1.2 → `dev-docs/project-brief.md` → `research/PAPER_POSITIONING.md`。

---

## 1. 任务目标（一句话）

搭建 Pattern Recognition 英文初稿骨架，完成**材料已齐备的章节**：Related Work（吃 W1 成果）+ Method 框架（吃双仓架构口径）+ Experiment 骨架（数字留占位符）。

**明确不做**：实验数字填充（等 P0.2-P0.5）、Introduction 终稿（等全管线结果后重写贡献列表）、投稿格式排版。

## 2. 执行链

1. 建 `docs/paper/` 目录，产出四件套：
   - `outline.md`——全文大纲（IMRaD + PR 期刊惯例），每节标注「已可写 / 待 P0.x 数据」
   - `related-work.md`——英文初稿，**锚定 W1 排出的三个最近邻写差距**：MAC-Learning（人类域锚点对比≠锚点种子聚类迭代）、Skeleton-to-Image（VFM→骨架表征≠小样本分类管线）、TP-CanineNet（动物 RGB 域伪标签≠骨架时序域）；证据源 `research/NOVELTY_CHECK_YAOQING_JIA.md` §3 排查表
   - `method.md`——英文初稿：物理-语义解耦架构（评估标准演化只改语义层）+ 六环管线（规则种子→AimCLR→SMQ→JIA迁移→TCL→主动学习→骨干微调）；素材 `project-brief.md` §1 + `research/RESEARCH_FOUNDATION_SYSTEM_VISION.md`
   - `experiment-skeleton.md`——三层指标口径表格骨架（合成/公开真实/真实 K9 三列，禁止混报），所有数字单元格写 `[TODO: P0.x]`
2. 全部英文撰写；中文思考过程写在每文件末尾 `<!-- CN-NOTES -->` 注释块（供用户评审）

## 3. 引用材料地图（全部只读）

| 材料 | 用途 |
|------|------|
| `research/NOVELTY_CHECK_YAOQING_JIA.md` | Related Work 差距分析 + 首次性主张措辞（注意保留「初步核验」限定语） |
| `research/RESEARCH_LITERATURE.md` | 动物行为识别全景引用池（BCST-GCN/ASBAR/PoseR…17 篇） |
| `research/PAPER_POSITIONING.md` | 标题/创新点/组件来源表 |
| `reports/p01-aimclr-2026-08-23.md` | Experiment 骨架中 P0.1 行可填真实数字（20.89% vs 8.33%） |
| `docs/DATA_LOCATIONS.md` | Datasets 节数据集描述（用实测值：AK 329 视频、APTv2 ~83K） |

## 4. 边界（与 W4/W6 并行互斥）

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `docs/paper/**`（全新目录，唯一 owner） |
| ❌ 禁触 | 一切代码/配置/`psd/`/`scripts/`/`configs/`/`external/`/`reports/`/`dev-docs/**`（只读引用除外）/`docs/DATA_LOCATIONS.md` |
| 环境 | **零 Python 依赖**——不要动 `.venv`（W4 正在使用，避免 pip 并发锁） |

## 5. 完成标准与 Git

- [ ] 四件套齐全，Related Work ≥800 词英文成稿
- [ ] 所有引用条目有真实出处（仓库名/DOI/arXiv 号），禁止编造文献——不确定的标 `[CITATION-NEEDED]`
- [ ] 提交：`feat: P0.6 论文初稿启动——大纲+Related Work+Method框架+Experiment骨架`；遇 `index.lock` 重试；禁 push
- [ ] 完成后在回复中请用户评审 outline.md 的故事线

## 6. 升级路径

发现 W1 结论不足以支撑某段 Related Work 表述 → 引用原文并在 CN-NOTES 标注疑问，不自行改判；故事线有重大分歧 → 停笔上报用户裁决。
