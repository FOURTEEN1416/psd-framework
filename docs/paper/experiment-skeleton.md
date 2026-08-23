# 4-5. Experiments & Ablation 骨架（三层口径 · P0.6）

> Owner: `docs/paper/experiment-skeleton.md` · W5 窗口 2026-08-23 · 状态: 骨架 v0.2——**全部数字 `[PENDING]` 占位，唯一例外为 P0.1 已归档实测值**
> v0.2 对抗评审加固：统计协议节 + 数据集引用义务注记 + NTU 实现正确性验证行【需用户决策】+ E1 折间方差披露。
> 规范来源: `analyze-results` 结果呈现规范 + galaxy 实验章要求（每个实验先声明支撑哪条 claim）+ AGENTS.md 三层口径铁律

---

## 4.1 数据集与三层口径协议（Datasets and the Three-Tier Protocol）

**铁律**：合成 / 公开真实 / 真实 K9 三层数字分列汇报，任何表格与正文禁止跨层混排（AGENTS.md 硬规则 3）。每个结果必须能回答"这是哪一层的数字"。

| 层 | 数据源 | 规模（2026-08-23 实测，`docs/DATA_LOCATIONS.md`） | 本文用途 |
|----|--------|------|---------|
| 合成层 | ST-GCN+BC / Mamba 合成管线（K9 移植） | ⏳ 待移植后登记 | 管线联调与消融快速迭代 |
| 公开真实层 | InterPet4D | 226 个 SMAL 拟合序列（225 有效，1 全帧 NaN 剔除） | 自监督预训练（C3/C8）+ 分割评估（C4） |
| 公开真实层 | Animal Kingdom 犬科 | 329 视频（train 231 / test 98）、34,772 帧行标注 | 弱监督/行为分类评估（C5/C6） |
| 公开真实层 | APTv2 | 83,304 文件 | 无标签池扩展 |
| 真实 K9 层 | 工作犬训练视频 | ⏳ 非 P0 启动硬依赖；主动学习目标域 | C7 终验 |

数据集描述段写作要点：各源采集场景、骨架维度（kp_world (T,24,3) 世界坐标 + 置信度通道）、许可证状态（开源终审保留项）、与人类域基准的本质差异。

## 4.2 实现细节（Implementation Details）

- 硬件：RTX 5060 Laptop 8GB 单卡；预训练 120 epoch 约 13 分钟（P0.1 实测）——论文如实报告消费级算力可行性。
- 协议：kNN 与微调均报 5-fold mean±std；主表另附随机基线与倍率。
- 种子：⏳ [PENDING] 固定种子清单与重复次数（目标 ≥3 seeds）。
- 超参冻结表：见 `method.md` §超参与复现清单；未冻结项在数据到位前不得写入正文。
- 复现性：一条命令复现序列已在 `reports/p01-aimclr-2026-08-23.md` §7 归档，投稿时整理为 appendix。

### 统计协议（v0.2 新增，PR 统计严谨性要件）

| 项 | 执行标准 |
|----|---------|
| 集中趋势与离散度 | 一律 mean±std（fold 级或 seed 级，注明来源）；禁止只报单次最优 |
| 显著性检验 | 主表对比做 fold 级配对检验（正态性通过用配对 t 检验，否则 Wilcoxon 符号秩），报告 p 值；基线对照同规则 |
| 多重比较 | 同表 ≥3 组对比时做校正（Holm-Bonferroni），注明校正方法 |
| 效应量 | 倍率/Δ 与原始值并列；kNN 类探针附随机基线参照线 |
| 异常值 | 显式列出并解释（如 P0.1 的全 NaN clip 剔除），禁止静默剔除 |
| 折间方差披露 | E1 现有折间极差大（15.56%–26.67%），成稿必须讨论方差来源（序列数少、个体差异）而非隐藏 |

> 数据集引用义务：§4.1 提及的 InterPet4D / Animal Kingdom / APTv2 必须引原始论文（outline §6 已挂三条 [CITATION-NEEDED]）。

## 4.3 主实验（Main Results）——逐实验声明 claim

### E1 物理层表征质量 → 支撑 C3
*This experiment tests whether self-supervised pretraining on unlabeled quadruped skeletons yields discriminative dynamics representations.*
- 口径：公开真实层（InterPet4D），dog-ID 代理探针（口径披露句随行）。
- **已可填**：kNN top-1 **20.89% ± [PENDING std]** vs 随机 8.33%，倍率 **2.51×**（fold 明细：20.00/15.56/26.67/22.22/20.00；⚠️ 折间极差 11.1 个百分点，成稿须披露并归因于 225 序列的小样本方差）。来源：`reports/p01-knn-result.json`。
- 对照行：projection-head 特征口径 12.89%（1.55×）——两口径并存披露，不择优单报。

### E2 无监督分割质量 → 支撑 C4
*This experiment tests whether motion-word quantization cuts continuous streams into behavior-aligned proposals.*
- 边界 IoU vs 滑窗基线：**[PENDING P0.2]**；fig3 定性可视化配对呈现。

### E3 锚点-伪标签扩展 → 支撑 C5
*This experiment tests whether a small rule-engine seed set expands to taxonomy coverage via iterated pseudo-labeling.*
- 聚类纯度 / 覆盖率随迭代曲线；种子比例扫描：**[PENDING P0.3]**。

### E4 半监督自训练 → 支撑 C6
*This experiment tests whether self-training closes the gap to full supervision under ≤20% labels.*
- 三层口径 × 标注比例 {10%, 20%, 100%} 主表 tab2：**[PENDING P0.4]**。
- 人类域参照行（明确标注为他域参照，不计入本文结论）：TCL CVPR 2021 82.7% @10% vs 全监督 88.6%。

### E5 主动学习效率 → 支撑 C7
*This experiment tests whether uncertainty sampling reaches the 22-class ≥85% target within a 100–200 clip human budget.*
- 效率曲线 fig4（vs 随机采样）：**[PENDING P0.5]**。

### E6 解耦切换成本 → 支撑 C1
*This experiment tests whether taxonomy evolution under decoupling costs less than full-pipeline retraining at matched accuracy.*
- 成本度量 = 人工标注单元数 + 重训墙钟时间；非解耦基线对照：**[PENDING P0.5]**。
- ⚠️ **Y′ 来源要求（v0.3）**：taxonomy 演化场景不得作者自造（防"稻草人演化"指控）——必须从真实业务依据推导（如 K9 业务评估标准变化记录）或采用可公开陈述的基准化协议，成稿需一段 Y′ 合理性论证。
- ✅ **双贴合候选场景已预注册并获用户方向性认可（ADR 0002 v1.1，2026-08-24）**：业务动机取自 K9 系统真实报表粒度差——日常训练日报（粗粒度：合并步态类为 locomotion）vs 结业考核单（细粒度：jump 拆分 jump_up/jump_down）；可计算基础取自本仓物理先验 7 类。论文叙事闭环：§1 行业动机 → §3.4 形式化 → E6 实例，同源非自造。P0.5 前用户一句话最终定稿。
- ⚠️ **matched accuracy 判据固定（v0.3）**：两侧使用相同训练预算与收敛判据（如固定 epoch + 验证集早停 patience 一致），禁止事后选择最优点制造有利对比。

## 4.4 对比研究（Comparison Studies）

| 对照轴 | 基线 | 状态 |
|--------|------|------|
| 预训练骨干 | AimCLR vs AimCLR++（PR 2024，77.2% NTU xsub 参照）vs 随机初始化 | ⏳ |
| 分割策略 | 运动词量化 vs 滑动窗口 vs 均匀切分 | ⏳ 待 P0.2 |
| 语义扩展 | 锚点聚类伪标签 vs 一致性正则 vs mean-teacher | ⏳ 待 P0.3/P0.4 |
| 动物域参照 | ASBAR（PoseConv3D）/ BCST-GCN 式图卷积（可复现行） | ⏳ 视工程量裁剪，砍项需用户确认 |
| **实现正确性验证**✅已批准（ADR 0002） | 在 NTU60 上复现 AimCLR 参照成绩（口径已于 W9 核实：NTU60 xsub 线性评估 + 三流融合；论文正文 78.9%，79.18% 为官方 released-model 复测——`reports/ntu-phasea-2026-08-24.md`；预注册通过线维持 ≥77.18%），证明本仓适配实现与官方等价后再用于动物域——重实现类论文的标配防线（风险登记册 R4）。**⚠️ 执行前置：本仓无 NTU 数据，Phase A 数据获取受阻于网络（GDrive 本机不可达，百度云镜像需账号交互），渠道裁决待用户（选项见报告 §4.3）；预处理版实际 ≈2GB** | 🔄 Phase A 进行中：协议核实 ✅ / 数据获取 🚧 待裁决；Phase B 等 P0.2 释放 GPU |

## 5. 消融与分析（Ablation and Analysis）— tab3 占位

| 开关 | 关闭后果假设 | 对应 claim |
|------|------------|-----------|
| − 自监督预训练 | kNN 掉回随机水平附近 | C3 |
| − 无监督分割 | 无法处理连续流 → 只评片段级 | C4 |
| − 锚点引导 | 聚类漂移、纯度下降 | C5 |
| − 伪标签迭代（单轮） | 覆盖率停滞 | C5 |
| − 主动学习（改随机采样） | 达标预算超 200 片段 | C7 |
| **种子噪声注入（v0.3 新增）** | 向规则引擎种子注入 {10%, 20%, 30%} 标签噪声，验证伪标签迭代不放大种子错误——语义层鲁棒性防线（对应 method.md §3.3.2 设计决策） | C5 |

敏感性扫描：种子比例 / 聚类数 K / 置信度阈值 τ / 迭代轮数 / 类别不平衡处理（frequency-aware margin vs 重采样 vs 无处理）。
分析段统一采用四步结构（observe→interpret→implicate→next），异常值显式标记不静默剔除。

## 图表规范（画图阶段生效，从「参考论文使用指南」62 篇统计规律英文期刊裁剪）

| 规则 | 执行标准 |
|------|---------|
| 底色与网格 | 白底 #FFFFFF + 浅灰网格线（#DADADA 系）；禁深色底 |
| 配色数量 | 单图数据系列色 ≤6；低饱和淡彩优先（参考淡红 #FFDADA / 淡青 #DAFFFF / 淡蓝 #DADAFF / 淡黄 #FFFFDA 族） |
| 色盲安全 | 禁纯红-纯绿组合；优先蓝-橙对；成图须过灰度打印检查 |
| 格式 | 曲线/柱状矢量输出（PDF/EPS）；照片类定性图 PNG ≥600 DPI；图内不放标题（caption 自足） |
| 表格 | 最优值加粗 + 方向符号（↑/↓）；数值右对齐、小数位一致 |
| 误差棒 | 多种子/多折必带误差棒并注明 std 或 SE |

## 未验证项与移交

- [ ] 所有 `[PENDING]` 数字待 P0.2-P0.5 报告归档后回填，回填时同步更新 outline.md 的 Claims-Evidence 矩阵状态
- [ ] E4 对比研究"动物域参照行"若工程量超支，砍项决定须上报用户
- [ ] **NTU 实现正确性验证行是否纳入 P0 范围——待用户决策（风险登记册 R4 的关键缓解项）**
- [ ] 合成层规模登记待 K9 资产移植（`docs/assets-map.md` 流程）

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 窗口骨架：E1-E6 实验-claim 映射 + 三层协议 + 图表规范 |
| v0.2 | 2026-08-23 | 对抗评审加固：统计协议（配对检验+多重校正）+ 折间方差披露 + NTU 验证行【需用户决策】+ 数据集引用义务 |
| v0.3 | 2026-08-23 | 第二轮对抗评审：E6 增 Y′ 来源要求与判据固定条款 + 消融增种子噪声注入行 + 敏感性增不平衡处理开关 |
