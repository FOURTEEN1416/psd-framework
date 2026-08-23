# 姚青 JIA 方法创新性核验报告（NOVELTY_CHECK）

> **执行**: W1 窗口（歆歆）· 2026-08-23
> **工具链**: GitHub MCP 强制调研（AGENTS.md 零容忍 WebSearch）✓
> **核验对象**: 「图像域 VFM 骨干 + 锚点学习 + 自监督聚类 + 伪标签迭代」迁移到**时序骨架域**的首次性主张
> **结论速览**: ✅ **首次性初步成立**（检索矩阵内未发现占坑工作；投稿前需 arXiv/Scholar 人工终审，见 §5 边界声明）

---

## 1. 判定基准（什么算占坑）

占坑 = 存在同时满足以下三要素的已发表工作/公开代码：
1. 特征来自视觉基础模型或其骨架域等价物（自监督骨架编码器）
2. 使用**少量人工标注锚点种子**引导
3. 通过**聚类产生伪标签 + 置信度筛选 + 迭代自训练**完成识别

单要素工作不算占坑（各组件均有文献）；few-shot 元学习 episode 范式 ≠ 本组合（无伪标签迭代闭环）；通用半监督一致性正则 ≠ 本组合（无锚点种子引导聚类）。

## 2. 检索矩阵（10 组查询全覆盖）

| # | 维度 | 查询词 | 命中 |
|---|------|--------|------|
| 1 | 直接命中 | `few-shot skeleton based action recognition` | 4 仓，全人类域元学习范式 |
| 2 | 直接命中 | `semi-supervised skeleton action recognition pseudo-label` | **0** |
| 3 | 伪标签自训练 | `skeleton action recognition pseudo-label self-training` | **0** |
| 4 | 动物骨架 | `animal behavior skeleton clustering unsupervised` | **0** |
| 5 | 动物骨架 | `animal pose behavior recognition semi-supervised self-training` | **0** |
| 6 | VFM 迁移 | `DINO vision foundation model skeleton action recognition` | **0** |
| 7 | AimCLR 生态 | `AimCLR contrastive skeleton` | 官方仓 Levigty/AimCLR（72★，2026-04 活跃）；无半监督扩展仓 |
| 8 | 兜底组合 | `skeleton clustering anchor pseudo-label few-shot self-training` | **0** |
| 9 | 代码级兜底 | code search `"pseudo label" anchor clustering skeleton action filename:README` | 37 条全为论文列表仓，无方法实现仓 |
| 10 | awesome 目录 | firework8/Awesome-Skeleton-based-Action-Recognition（727★，月更至 2026-08）全量扫描 2014-2026 | 19 个候选条目逐一排查 |

## 3. 候选工作排查表（≥8 项逐一验证）

| 候选 | 出处/活跃度 | 与本组合的差异 | 占坑? |
|------|------------|---------------|-------|
| MAC-Learning | TPAMI 2022，[1xbq1/MAC-Learning] | NTU/UCLA/Kinetics 人类域；"锚点"=对比学习锚样本；无聚类伪标签迭代、无 VFM | ❌ |
| Momentum Contrastive Teacher | TIP 2025 | 人类域半监督一致性范式（mean-teacher 变体） | ❌ |
| GRA (Graph Representation Alignment) | IEEE | 人类域半监督图对齐 | ❌ |
| PSP-Learning / X-Invariant / Decouple-and-Squeeze / Joint-bone Fusion | 2022 半监督系列 | 人类域一致性/对比范式，NTU 口径 | ❌ |
| PAINet (ICCV 2023, 11★) | starrycos/PAINet | few-shot 元学习 episode 分类，无伪标签迭代 | ❌ |
| HAA4D (10★) / ISBFSAR (14★) / FICAMA (1★) / SMAM / UMEG-Net / SkelHCC | 2021-2026 | 同上，元学习范式，人类域 | ❌ |
| Skeleton-to-Image Encoding via Vision-Pretrained Models | arXiv 2603.05963 | VFM 用于骨架表征学习（骨架转图像），**不是**小样本锚点+聚类伪标签分类管线 | ❌（最接近项之一，Related Work 必引） |
| TP-CanineNet | MDPI Animals 2025，无代码 | 犬行为**RGB 视频域**伪标签；非骨架时序域、无锚点聚类组合 | ❌（动物域最接近项，必引） |
| 无监督骨架分割：Skeleton Motion Words (ICCV 2025) / Hierarchical ST-VQ (arXiv 2604.15196) | — | 相邻方向但目标是无监督**分割**，非锚点引导小样本**识别** | ❌ |

## 4. 结论与证据强度

**✅ 首次性初步成立**：

1. 组合词检索在 GitHub 全网零命中（查询 2/3/4/5/6/8 六组全 0）
2. 代码级全文搜索无任何方法实现仓（查询 9）
3. 最权威 awesome 目录（727★ 月更）2014-2026 全量扫描后，所有 semi-/few-/anchor- 相关条目均为人类域且范式不符
4. 动物骨架域连通用半监督工作都未检出（空白区比预想更大——对论文是利好）
5. AimCLR 生态内无半监督/伪标签扩展实现

**对 P0.3 与论文框架的影响**: 无需重新设计。建议 Related Work 锚定三个最近邻写差距：MAC-Learning（人类域锚点对比）、Skeleton-to-Image（VFM→骨架）、TP-CanineNet（动物域伪标签）。

## 5. 边界声明（诚实版）

- ⚠️ **未验证面**: GitHub 工具链对「有论文无代码」的工作覆盖有限（AGENTS.md 禁 WebSearch 的结构性代价）。**投稿前必须做一次 arXiv / Google Scholar 人工终审**，本报告不替代它。
- ✅ ~~niais/Awesome 目录未逐条扫描~~ **已于同日补扫确认**：`Yingfei-Wu/Awesome-Self-supervised-Skeleton-based-Action-Recognition` 全量 43 条目均为自监督表征学习范式（MSM/CL，NTU 线性探针协议），零「锚点+聚类伪标签」组合、零动物域——结论不变。
- 📌 **附带发现**: AimCLR 官方续作 AimCLR++ 存在（`Levigty/AimCLR-v2`，PR 2024，NTU xsub 77.2% vs v1 的 74.3%）——P0.1 预训练骨干若收敛不佳，可直接升级候选。
- 中文文献（知网系）完全不在覆盖范围。

## 6. 建议回写清单（由用户或收尾会话统一执行，W1 未越权改动）

| 文档 | 位置 | 建议 |
|------|------|------|
| `research/PAPER_POSITIONING.md` | L12「首次性待核验」 | →「GitHub 链初步核验通过（见 NOVELTY_CHECK），投稿前 Scholar 终审」 |
| `dev-docs/project-brief.md` | §8 待确认事项第 1 条 | 标记初步完成，保留终审项 |
| `dev-docs/stage-plan.md` | 启动前置 #1 | 勾选 ✅（保留 Scholar 终审备注） |

## 7. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-23 | W1 窗口完成 10 组查询 + 14 项候选排查，结论：首次性初步成立 |
