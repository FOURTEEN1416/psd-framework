# 4-5. Experiments & Ablation 骨架（三层口径 · P0.6）

> Owner: `docs/paper/experiment-skeleton.md` · W5 窗口 2026-08-23 · 状态: **终稿候选 v2.0（2026-09-05 R16/R17 端到端协议诚信修正轮——E7/E7b/E9 端到端数字以修正协议工件为准，本节原文保留为历史归档；前序 v1.3（2026-09-04 P0.8 语义warm-start消融 + K9 试点预注册轮；v1.2 = 2026-09-02 协调者三流收官收编轮；v1.1 = W45 预训练价值梯度轮）**——E1-E4 数字在档；E5 负结果如实入册且 C7 换轨定稿；E6 成本 6.07× 背书 ≥3×；tab3 七行全部映射溯源完成；**§4.4 NTU 行 R4 ✅ 终判 PASS（2026-09-02）：三流 pretext 300/300 ×3 + LE joint 74.30 / bone 71.51 / motion 67.84 + 3s 融合 77.97% ≥ 预注册线 77.18%（top5 95.78%, n=16487）——本仓适配实现与官方等价性成立，风险登记册 R4 解除**；v1.1 增补（W45）：tab3 梯度叙事 + 低资源梯度段 + L10 前送（保持不变）；v1.3 增补：tab3 新增 −语义warm-start 行（P0.8 同协议 AimCLR 对比臂，top-1 全预算占优/macro-F1 混合如实并报）+ K9 真实域试点预注册协议（`docs/paper/k9-pilot-preregistration.md`，PSD-K9-PREREG-001，数据授权前冻结）
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
⚠️ **真实域有效监督口径（2026-08-25 新增，C5/W29 发现）**：dog-pose GT 从未标注 idx20-23（双眼/withers/throat——StanfordExt24 后四点不在 StanfordExtra 源数据，BARC data_info 原文佐证），故**公开真实层提点产物为拓扑同构、有效监督 20/24**；提取管线已在组装出口硬掩码该四通道（DEAD_JOINTS 机制 + NaN 化入规则引擎），正文与图表涉及真实域骨架时统一采用 20/24 表述并进 limitations。

## 4.2 实现细节（Implementation Details）

- 硬件：RTX 5060 Laptop 8GB 单卡；预训练 120 epoch 约 13 分钟（P0.1 实测）——论文如实报告消费级算力可行性。
- 协议：kNN 与微调均报 5-fold mean±std；主表另附随机基线与倍率。
- 种子：✅ **种子清单已落定（W36 终填，逐实验溯源）**——P0.1 kNN：固定 seed=42 + 5-fold CV（`reports/p01-knn-result.json`）；P0.3 原型聚类：3 run_seeds 纯度同值 0.5339（`reports/p03-jia-phasea-results.json`）；W14/W23 AL：两臂各 3 seeds（42/43/44），曲线带 mean±std；C1 解耦成本：small/full 两档同 seed 配对（同 seed 配对最小比值 4.00×，full 档）；Q3c 公开真实微调：seed=42 单次（`reports/p05-public-real-partialclass-result-2026-08-25.json` config_echo.seed）；E-C SMQ：单 checkpoint 多 episode（std 来自 episode 抽样而非种子重复）。⚠️ 缺口如实披露：P0.4 主操作点与 Q3c 为单种子结果，成稿引用时按 §统计协议节收窄措辞或投稿前补多种子复跑。
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

> 数据集引用义务：§4.1 提及的 InterPet4D / Animal Kingdom / APTv2 必须引原始论文——✅ 已于 W17 文献终审补齐官方题录（见 outline §6：Peng et al. 2026 dataset / Ng et al. CVPR 2022 / Yang et al. arXiv 2312.15612）。

## 4.3 主实验（Main Results）——逐实验声明 claim

### E1 物理层表征质量 → 支撑 C3
*This experiment tests whether self-supervised pretraining on unlabeled quadruped skeletons yields discriminative dynamics representations.*
- 口径：公开真实层（InterPet4D），dog-ID 代理探针（口径披露句随行）。
- **已可填**：kNN top-1 **20.89% ± 4.04% (公开真实层/P0.1；fold 级样本 std ddof=1，R13 对账修正——原 4.45 系历史 quickref 错误)** vs 随机 8.33%，倍率 **2.51×**（fold 明细：20.00/15.56/26.67/22.22/20.00；⚠️ 折间极差 11.1 个百分点，成稿须披露并归因于 225 序列的小样本方差）。来源：`reports/p01-knn-result.json`。
- 对照行：projection-head 特征口径 12.89%（1.55×）——两口径并存披露，不择优单报。

### E2 无监督分割质量 → 支撑 C4
*This experiment tests whether motion-word quantization cuts continuous streams into behavior-aligned proposals.*
- 边界 IoU vs **等段数随机切分基线**（蒙特卡洛 null 对照，`psd/training/segment_iou.py::random_baseline_mean_iou`，等段数随机切割 200 次模拟期望）：**0.458 ± 0.049 (公开真实层/P0.2 E-C)**；fig3 定性可视化配对呈现。
- ⚠️ 基线标注勘误（W34 排查定案，2026-08-25）：此处对照是**随机切分 null**，不是滑窗基线——全仓不存在滑动窗口/均匀切分方法臂（W7 任务书中"滑动窗口+相似度分割"仅为未执行的预案）。论文成稿禁止写成 "vs sliding window"；若需该对照臂见 tab3 −无监督分割行的新实验设计。

### E3 锚点-伪标签扩展 → 支撑 C5
*This experiment tests whether a small rule-engine seed set expands to taxonomy coverage via iterated pseudo-labeling.*
- 聚类纯度 / 覆盖率随迭代曲线；种子比例扫描：**0.503 ± 0.006 (公开真实层/P0.3)**。

### E4 半监督自训练 → 支撑 C6
*This experiment tests whether self-training closes the gap to full supervision under ≤20% labels.*
- 三层口径 × 标注比例 {10%, 20%, 100%} 主表 tab2：**0.691 ± 0.013 (公开真实层/P0.4)**。
- ✅ **tab2 公开真实列第二行（W36 终填，relay Q3c 落地数字）**：AK partialclass4 冻结骨干 + 头部重训（warm-init from Y best.pt）→ best_val_acc **44.90%（=1.80× 随机基线 25%）**【公开真实层/4 类部分口径】；per-class watch **100%** / track **23.5%** / jump **0%** / stay **0%**——类不平衡主导聚合分数（train 支持 watch 72/track 46/stay 27/jump 27），成稿必须逐类分解同框呈现并前送 Limitations L7，禁止聚合单报。来源：`reports/p05-public-real-partialclass-result-2026-08-25.json`（seed=42 单次，26ep 早停 @11）+ 协议链 `reports/p05-public-real-partialclass-2026-08-24.md`。⚠️ 扩展池增强第二轮（round2）独立窗口执行中，结果落地前本行不得预写"飞轮有效"结论。
- 人类域参照行：❌ 已删除（2026-09-04 R8 引用实查：TCL arXiv 2102.02751 原文 15 页 PDF 检索 NTU=0 命中、82.7=0、88.6=0，其数据集为 Mini-Something-V2/Jester/Kinetics-400/Charades-Ego——该数字系池内继承的无出处假引用，正文/摘要/outline 全链清除；bib 作者同步修正 Ankit Singh…Abir Das）。

### E5 主动学习效率【C7 已裁决①（2026-08-25）：降级"探索性发现"，不作主结果主张】
*This experiment examines how uncertainty sampling behaves under cold-start weak scorers within a limited annotation budget (exploratory; negative result reported as-is).*
- **已可填（负结果如实，合成层短预算档/W14）**：冷启动弱打分器场景下**熵采样未显示优势**——b=100 随机反超 **+7.9pp**、b=200 随机反超 **+7.1pp**（3/3 seeds 同向）；低预算区（b≤50）两臂均停在随机水平。来源：`reports/w14-p05-al-efficiency-2026-08-24.md` §3.1 + 曲线 JSON `reports/p05-al-efficiency-short-2026-08-24.json`。
- **真实池打分诊断（公开真实层排序清单，非精度数字）**：best.pt 在 P0.4 移交池上 softmax 全饱和（logit top1−top2 边际 mean=100.9），熵信号数值零退化，Top-K 清单不可行动；跨域佐证——同一模型在合成验证集边际均值仅 ≈10.8（域漂移 ~10×）。来源：同报告 §3.2。
- full-budget 档 `configs/p05_al_full.yaml` 已备好待 GPU 队列复跑；结论稳定性以复跑为准。
- ✅ **C7 措辞裁决记录（用户拍板选项①，2026-08-25，采纳歆歆推荐）**：C7 从贡献主张降级——负结果以"探索性发现"形式写入 §5 分析节（保留科学叙事价值：不确定性采样前提 = 较强打分器 + 域内校准 → 强化"先标注→校准打分器→再选样"渐进标注叙事）；"不确定性采样使 100–200 片段达到 ≥85%"这一效率主张在正文与摘要中禁用。
- 升级通道保留：W23 warm-start（裁决 A2）或冷启动 full-budget 复跑产出正证据 → 经用户新裁决后方可恢复 C7 主张地位并回填。
- fig4 引用保留：负结果效率曲线照画（W22 已绘制），落点随降级从 §4.3 迁至 §5 分析节。
- ✅ **E5 叙事换轨（用户裁决 A，2026-08-25 晨，采纳歆歆推荐）**：E5 正证据路径由"采样效率"换轨为 **"warm-start 语义层初始化使小标注预算可用"**——合成偏移层 **82.0%@20 片段**（冷启动同预算仅 ~7.8%），b=200 天花板 95.7%；两轮独立执行逐字节一致（W23 worktree 规范跑 + 主检出复算）。来源：`reports/w23-p05-al-warmstart-2026-08-25.md` §1/§3.1 + JSON `reports/p05-al-efficiency-warmstart-short-2026-08-25.json`。口径：合成偏移层（noise_std=0.10），三层铁律照旧。
- 效率主张正式关闭：强打分器下熵仍负（随机 3/3 seeds × 3/3 预算点反超 4.2~5.0pp）→ "难例优先 vs 均匀覆盖"作为边界条件探索性发现进 §5；**类间轮转熵等混合策略一句话 future work，不再开实验线**。
- ~~Q4 full-budget 方向稳定性终验~~ ❌ 用户裁决同晨叫停省时；如未来需要由新任务书重启。

### E6 解耦切换成本 → 支撑 C1
*This experiment tests whether taxonomy evolution under decoupling costs less than full-pipeline retraining at matched accuracy.*
- 成本度量 = 人工标注单元数 + 重训墙钟时间；非解耦基线对照。**已可填并经 full 档确认（合成层/W19 C1 实验 + relay Q2 full 档，2026-08-25 W34 按"以 full 为准"回改）**：
  - 墙钟比：small **7.32×**（80s vs 588s，CPU）→ full **6.07×**（188.7s vs 31.1s，干净 GPU 路径 cuda+AMP）；论文措辞采用保守区间 **≥3×**——原 CPU 负载污染疑虑已被 full GPU 复跑正面消除；同 seed 配对比值最小 4.00×、最不利裁剪口径 3.50×，保守下界稳
  - 精度差：full **−0.91pp**（96.67% vs 95.76%）/ small **+2.27pp**（96.0% vs 93.7%）——两档方向相反但幅度均 <2.3pp（full val n=440 时 1 样本≈0.23pp），措辞定为 **统计等效、无显著精度代价**（⚠️ 禁用"无精度代价且略胜"/"+2.27pp 略胜"等方向性表述）
  - 收敛轮数比：small 1.55× / full **2.18×**——第二成本维度同向佐证
  - 标注单元数两臂打平（small 528 train / 132 val；full 1760 train / 440 val）——成本优势来自算力而非标注节约，如实呈现不粉饰
  - 来源：`reports/c1-decouple-cost-2026-08-24.md` §2 + §9（full 档增补章节）；当次运行 JSON `reports/c1-decouple-cost-2026-08-24.json` + `reports/c1-decouple-cost-full-2026-08-25.json`
  - 口径边界：合成层单一演化类型（标签合并 Y→Y′）；预注册条款"趋势矛盾以 full 为准回改"**已执行完毕**（成本主张成立，精度措辞校准为统计等效）
- ✅ **R2 状态注记（W34 更新）**：成本维度已获 full 档背书（6.07× 干净 GPU 路径）→ 风险登记册 R2 🟡 **可转 ✅**——标题副句 *under Evolving Evaluation Criteria* 实证立场成立；outline.md 风险行的正式改判与终裁归协调者/用户。
- Y′ 合理性论证段素材已可成文（v0.3 要求落地）：ADR 0002 v1.1 预注册的 K9 报表粒度差业务动机——日常训练日报（粗粒度 locomotion 合并）vs 结业考核单（细粒度 jump 拆分 jump_up/jump_down），底稿见 outline §8 R10。
- ⚠️ **Y′ 来源要求（v0.3）**：taxonomy 演化场景不得作者自造（防"稻草人演化"指控）——必须从真实业务依据推导（如 K9 业务评估标准变化记录）或采用可公开陈述的基准化协议，成稿需一段 Y′ 合理性论证。
- ✅ **双贴合候选场景已预注册并获用户方向性认可（ADR 0002 v1.1，2026-08-24）**：业务动机取自 K9 系统真实报表粒度差——日常训练日报（粗粒度：合并步态类为 locomotion）vs 结业考核单（细粒度：jump 拆分 jump_up/jump_down）；可计算基础取自本仓物理先验 7 类。论文叙事闭环：§1 行业动机 → §3.4 形式化 → E6 实例，同源非自造。P0.5 前用户一句话最终定稿。
- ⚠️ **matched accuracy 判据固定（v0.3）**：两侧使用相同训练预算与收敛判据（如固定 epoch + 验证集早停 patience 一致），禁止事后选择最优点制造有利对比。

### E7 端到端管线有效性 → 支撑 C6（[R16 修正：本段"端到端量化"主张已撤回为负边界，见上方修订注] 低资源主张的端到端量化，冲击 PR 补强，2026-09-04 预注册 fb1a060）

> **⚠️ R16 协议修正（2026-09-05）**：本节原报告臂的最终分类器在**池片段真标签**上重训（且 precision-drop 停止消费真标签）——"13% 标注达 94%"与实现不符，系对抗审稿实锤的协议错误。修正协议（最终头=种子真标签∪池伪标签；停止=收敛/预算，无 oracle）重跑结果：warm spc2 **9.8%±7.5**（chance 11.1%）、aimclr 15.2%±5.1、scratch 8.4%±10.7——AK 层端到端近随机，**低资源端到端主张在本层撤回为负边界发现**；head_calib 诊断（GT 无关逐轮再校准）不救（9–12%）。全监督参照 33.93% 为纯监督臂，不受影响。证据 `reports/r16-endtoend-pseudo-2026-09-05.json`；驱动 `scripts/run_r16_endtoend_pseudo.py`。下文原文保留为历史归档。
>
> **P1 标签对齐修正（2026-09-05，PSD-ALIGN-PREREG-001，NULL）**：针对本负边界的头路过拟合坍缩，预注册三 GT 无关对齐变体（V1 全类锚点共识门/V2 配额/V3 原型路主导）+V0 对照重跑 AK v1 spc2×10seeds——最优 V2 14.11%±6.65（对 V0 配对胜 8/10、p=0.13）、V3 池精度 0.21 vs V0 0.11，机制方向有效但**未达 20% 救援线，判据 NULL**；约束=种子锚定特征几何（r0 原型精度 ~0.30 封顶），非门控规则。证据 `reports/p15-label-alignment-2026-09-05.json`；驱动 `scripts/run_p15_align.py`；§5 消融段已入文。
>
> **P2/P3 并行线（2026-09-05，ADR 0008，均 NULL/膝点未达）**：P2 APTv2 域自适应预训练骨干触发灾难遗忘（全监督 33.93→25%），V2+DAP 11.43%<P1-V2；P3 种子预算曲线 spc2→spc12（13%→77% 标注）仅 14.11%→22.14%，膝点（24%）全程未达，池伪标签净损害最终头。三独立杠杆（门控/骨干/预算）一致否证 AK 层端到端低资源可救援，约束=数据质量本身。证据 `reports/p16-dap-aptv2-2026-09-05.json` + `reports/p17-budget-curve-2026-09-05.json`。
*This experiment tests whether the complete PSD pipeline, run end-to-end from a small annotation budget, approaches full-supervision accuracy on the public-real tier.*
- 管线：Y/scratch backbone 特征 → 种子（每类 spc 个 clip）→ run_selftrain（锚点+原型聚类+置信过滤伪标签迭代）→ seeds∪pool 线性头 → val top-1。防泄漏：anchor_mask 与池宇宙严格限 train split，val(56) 全程隔离。
- **端到端结果（公开真实层 AK full12，9 类有样本，3 seeds mean±std）**：
  - warm spc2（≈13% 标注）：top-1 **0.3095±0.0223**，macro-F1 0.1244
  - warm spc4（≈26%）：top-1 0.3214±0.0253，macro-F1 0.1422
  - warm full（100% 全监督对照）：top-1 0.3393，macro-F1 0.1465
  - scratch spc2（随机 backbone）：top-1 0.2500，macro-F1 0.0451
- **两条主张**：① **低资源端到端成立**——13% 标注预算达全监督的 **91%**（0.31 vs 0.34，仅差 3pp），把"低资源"从组件级证据升级为端到端量化；② **预训练端到端价值**——warm vs scratch 同预算 +6pp top-1、macro-F1 **2.8×**（0.124 vs 0.045），少数类感知显著。
- **诚实边界**：绝对天花板 ~34% 由该层自提取数据瓶颈主导（197 clips/单标签/类不平衡，见 L7），非模型选择——与 §4.4 五臂窄带结论一致。
- 来源：`reports/p07-endtoend-ak-full12-2026-09-04.json`；驱动 `scripts/run_p07_endtoend_ak.py`。
- **⬆️ P1.0 种子扩容 supersession（2026-09-04 同日，n=3→10，论文主数字改用此口径）**：warm spc2 **0.3196±0.0245**（占全监督 **94%**）/ warm spc4 0.3143±0.0423 / scratch spc2 0.2464±0.0164（每 seed 独立随机骨干，torch 种子可复现）；配对 Wilcoxon：warm vs scratch top-1 **+7.32pp 10-0-0 p=0.002 显著**、macro-F1 +9.97pp 10-0-0 p=0.002 显著。p07 JSON 保留为 n=3 历史归档。来源：`reports/p10-seedexpansion-2026-09-04.md` + `.json`；驱动 `scripts/run_p10_seedexpansion.py`。

### E7b AK v2 预注册复现层 → 检验 E7 天花板归因（PSD-AKV2-PREREG-001，2026-09-04 构建前冻结）

> **⚠️ R16 协议修正（2026-09-05）**：EP2 低预算臂同受真标签头协议错误影响，修正重跑后 v2 warm spc2 **13.1%±5.3** / aimclr 17.6%±4.3 / scratch 13.8%±5.6（chance 12.5%）——88.6% 保留率与 warm>SSL 增强撤回；EP3 纯监督臂不受影响，但原始 +3.57pp 端点披露求解器路径噪声（标签重编码使 v2 full 37.50→35.42，2 个 val clip），归因锚定 §7 同空间对照 +11.54pp。协议修订见 `ak-v2-expansion-preregistration.md` §8。
*This experiment tests the data-bottleneck attribution of E7 under a frozen decision rule.*
- 构建：同 329 犬科视频多段重提取（K≤4 连续段、段≥40 帧、逐帧标签一致性门 ≥0.80、同 YOLO/assemble/质量门、视频级 split 不变）→ **352 clips（train 256/val 96），8 类空间**（bark 零段过门、sit n=2 如实披露）。驱动 `scripts/run_p11_ak_v2_build.py`，产物 `runs/public_real_dataset/full12v2_*`。
- **EP3 天花板检验（判据冻结：≥+3.0pp=数据瓶颈成立）**：v2 full **37.50%** − v1 33.93% = **+3.57pp**，原始判据触发；**R8 修订**：该差值被类别空间混淆（12→8 类 chance 位移 +4.17pp > +3.57pp），补同空间对照——v1 full 在 v2 的 8 类子空间重算=**25.96%**，同空间差 **+11.54pp**（above-chance 13.5→25.0pp、above-majority −4.8→+3.1pp）→ 数据瓶颈归因在同空间对照下于 top-1 成立；macro-F1 反向（14.7→7.8，sit n=2 驱动）双指标并报。**[R17 dated 修正 2026-09-05]**：上述 macro-F1 对值系评估器类索引缺陷（`range(len(class_names))` 遇非连续 int 标签丢越界类）产物——修正评估器下 v1→v2 full macro-F1 为 16.5%→19.3%，与 top-1 同向上升，"反向"撤回；见 `reports/p12-akv2-replication-2026-09-04.md` 勘误与 `r16-endtoend-pseudo` 工件 warm_full 行。
- **EP2 复现+增强（n=10）**：warm spc2 **33.23±3.35**（保留自身全监督 88.6%，绝对预算 2 片段/类=6% 标注比例）；warm vs aimclr top-1 **+12.92pp 10-0-0 p=0.002** 且 **macro-F1 +6.08pp 10-0-0 p=0.002**（v1 的 macro-F1 持平判定为单片段标签结构伪影——一致性门剔除混标段后 AimCLR 少数类预测优势消失: macro-F1 坍缩至 1.71%）；warm vs scratch top-1 +9.79pp p=0.002。
- 纪律：v1 数字不替换，v2 并列报告禁合并；v2 措辞用 matched absolute budget 禁写 13%。来源：`reports/p12-akv2-replication-2026-09-04.md` + `.json`；驱动 `scripts/run_p12_ak_v2_replicate.py`。
- **P1.3 端到端微调诊断对照（非预注册终点，control 措辞）**：v2 上解冻骨干端到端 50ep——finetune_full **32.99±3.94 < 冻结头 37.50**（256 段喂不饱骨干梯度路径，过拟合吞掉 pretext 先验）；finetune_spc2 **9.72±1.59** vs warm-start 33.23 = **3.4×**——"天花板非冻结所致"与"解耦设计实证更优"两条防御句的直接数字。来源：`reports/p13-v2-finetune-2026-09-04.md`；驱动 `scripts/run_p13_v2_finetune.py`。

### E9 NTU 低资源保留率 → 支撑 C6 跨域泛化（PSD-NTU-PREREG-001，2026-09-04 实验前冻结）

> **⚠️ R16 协议修正（2026-09-05）**：臂 (b) 修正重跑（伪标签最终头+无 oracle 停止）：top1 **67.5%±0.15**，保留率 **90.6%**——**预注册 ≥90% GENERALIZES 判据仍成立**；伪标签迭代增益 +1.4pp（原 +8.1pp 系真标签头产物）；池对官方标签精度 ≈0.69 仅作事后诊断。协议修订见 `ntu-lowres-preregistration.md` §7。证据 `reports/r16-ntu-pseudo-2026-09-05.json`。
*This experiment tests whether the low-resource retention behavior generalizes to the human domain where a published budget reference exists.*
- 设置: NTU60 xsub joint 流，冻结 epoch300 pretext 骨干 256d 特征（官方 Feeder_single 无增强口径，p14a 导出 40091+16487）；10% 分层子集 seed42 固定=4009 clips；三臂 (c)100% 线性头参照 /(a)10% 线性头 /(b)10%+run_selftrain（与 E7 同函数同参，适配 #1 device=cuda、#2 StandardScaler+LR tol=1e-3 披露随行）。
- **结果**: (c) **74.45%**（vs 官方 LE 74.30%，Δ0.15pp——管线保真自检通过）；(a) 66.05%（保留 88.7%）；(b) **74.01/74.06/74.25，mean 74.11±0.13 = 保留率 99.5%**。
- **预注册判据: GENERALIZES（≥90% 线）**。~~对照 TCL 发表保留率 93.3%~~（R8 引用实查证伪，对比删除——保留率仅与本预注册 90% 线比，不与任何发表方法比；且保留率轴协议依赖：冻结探针曲线天然平缓于细调管线）。次终点 (a)vs(b): 伪标签迭代人体规模贡献 **+8.1pp**（74.11−66.05=8.06），标注子集为动物层训练池 28×（4009 vs 141）。
- 诚实边界: 池 ~34.4k/seed（近恢复全训练集即机制本身）；NTU 无伪 GT 故池精度未单独评估——主张限于预算保留行为。
- 来源: `reports/p14-ntu-lowres-2026-09-04.md` + `.json`；驱动 `scripts/run_p14_ntu_featuredump.py` + `run_p14_ntu_lowres.py`；协议 `docs/paper/ntu-lowres-preregistration.md`。

### E9b NTU120 跨域第二点（HRNet 2D）→ 支撑 C6（PSD-NTU120-PREREG-001，2026-09-05 实验前冻结；ADR 0009）

> **⚠️ 首跑缺陷披露（修复先于读数）**：首轮 b 臂评分将字符串预测与整数标签比较（恒 False），伪记 b=0.0/FAILS；经"精确零不可能"检查定位，修为字符串域评分后重跑——臂/预算/种子/判据零改动，重跑为首次有效读数。协议偏离（pretext 由冻结的 ST-GCN 150ep 改为 joint-level MLP 80ep，适配 17 关节 HRNet 拓扑跨数据集一致）已入协议 Amendment 2。
- **结果（n=12000 clips，60 类有样本；10-seed 终口径 seeds 42–51，E9 系列扩容先例，2026-09-07）**: (c) 全预算参照 **54.34%**；(a) 10% 线性 **45.42%**（线性保留 83.58%）；(b) PSD 语义管线 **mean 48.29% ± 0.76 = 保留率 88.86% → 预注册判据 PARTIAL（85–90% 带）**；PSD 臂对裸线性 +2.9pp（10/10 种子臂高于线性臂）。3-seed 初读（48.36/49.65/47.48 → 89.24% PARTIAL）留档协议文件，判据/臂/预算零改动。
- 解读: 犬层导出的语义管线在第二个人体基准上保留率落入 PARTIAL 带，弱于 NTU60 的 90.6% 但仍支持预算保留行为；弱化与参照臂质量衰减（74.3→54.3）同向。
- 来源: `reports/p5b-ntu120-retention-2026-09-07.json`（10-seed 终口径；09-05 3-seed 文件留档）；驱动 `scripts/run_p5b_generic_retention.py`；协议 `docs/paper/ntu120-preregistration.md`（Amendment 3）。

### E9c UCF101 独立第三域（HRNet 2D, YouTube）→ 预算行为负边界（PSD-UCF101-PREREG-001，2026-09-05 实验前冻结；ADR 0009）

- **结果（n=10000 clips，101 类；10-seed 终口径 seeds 42–51，E9 系列扩容先例，2026-09-07）**: (c) 全预算参照 **23.11%**；(a) 10% 线性 **14.04%**（线性保留 60.75%）；(b) PSD 语义管线 **mean 15.40% ± 1.89 = 保留率 66.64% → 预注册判据 FAILS（<85%）**；PSD 臂对裸线性 +1.4pp（2/10 种子低于线性臂，不做同向声明）。3-seed 初读（12.74/14.47/16.20 → 62.62% FAILS）留档协议文件，判据/臂/预算零改动，方向未翻转。
- 诚实上报（按冻结判据）: 预算保留行为随域收窄，E9 主张限于其已测域。跨域梯度 NTU60 90.6% → NTU120 88.9% → UCF101 66.6% 与参照质量 74.3% → 54.3% → 23.1% 单调对应——**pretext 特征质量是保留率的主导因素**；与 R16 犬科负边界（warm-start 增益以"pretext 学到类可分特征"为前提）同构，入限制段与负边界叙事。
- 来源: `reports/p5b-ucf101-retention-2026-09-07.json`（10-seed 终口径；09-05 3-seed 文件留档）；驱动 `scripts/run_p5b_generic_retention.py`；协议 `docs/paper/ucf101-preregistration.md`（Amendment 2）。

### E9d PanAf500 首个动物域公共基准（P7，YOLO11x-pose 模型提取骨架）→ C6 跨域第四点（PSD-PANAF-PREREG-001，2026-09-06 实验前冻结；ADR 0009）

> **⚠️ 判据翻转披露（种子扩容，如实勘误）**：3-seed 初读 b=60/56/52 → **retention 93.33% → CONFIRMS**（ledger v2.2，commit 8ddb3d1）；按 E9 系列惯例扩至 10 seeds（42–51）后 **b mean 53.2% → retention 88.67% → PARTIAL**（10-seed 终口径，commit ca7ffa6/abac909）。判据/臂/预算/pretext 零改动；3-seed CONFIRMS 留档 git 史，被 10-seed PARTIAL 取代为上报口径。
- **结果（n=500 clips，9 类，10-seed 终口径 seeds 42–51，2026-09-07）**: (c) 全预算参照 **60.0%**；(a) 10% 线性 **52.0%**；(b) PSD 语义管线 **mean 53.2% ± 9.1 = 保留率 88.67% → 预注册判据 PARTIAL（85–90% 带）**。test split（75 clips，判据外次级）56%。P2′ 附带确认：POOL 表（det 0.821）提取漏斗 `reports/p7-extract-quality.json`。
- 披露: 9 类长尾（三类 ≤7 clips；10% 预算下五类单种子 clip）放大种子方差；帧级标注多数投票至 clip 级（与犬科层同粗粒度，如实披露而非修复——E7c 归因已收窄为犬科层特有协议）。
- 来源: `reports/p23-panaf-retention-2026-09-07.json`；驱动 `scripts/run_p23_panaf_retention.py`；协议 `docs/paper/panaf-preregistration.md`（Amendment 2026-09-07）。

## 4.4 对比研究（Comparison Studies）

| 对照轴 | 基线 | 状态 |
|--------|------|------|
| 预训练骨干 | AimCLR vs AimCLR++（PR 2024，77.2% NTU xsub 参照）vs 随机初始化 | ⏳ |
| 分割策略 | 运动词量化 vs 滑动窗口 vs 均匀切分 | ✅ **三臂对照已落地（W38，2026-08-26）**：SMQ 0.458±0.049 > 等段数均匀滑窗 0.399±0.035（预注册判据通过）> 随机切分 null 0.323±0.022；固定网格旁证臂 0.453±0.027 与 SMQ 统计等效如实披露——完整消融证据与双向论证见 tab3 −无监督分割行 |
| 语义扩展 | 锚点聚类伪标签 vs 一致性正则 vs mean-teacher | ⏳ 待 P0.3/P0.4 |
| 动物域参照 | ASBAR（PoseConv3D）/ BCST-GCN 式图卷积（可复现行） | ⏳ 视工程量裁剪，砍项需用户确认 |
| **实现正确性验证**✅已批准（ADR 0002） | 在 NTU60 上复现 AimCLR 参照成绩（口径已于 W9 核实：NTU60 xsub 线性评估 + 三流融合；论文正文 78.9%，79.18% 为官方 released-model 复测——`reports/ntu-phasea-2026-08-24.md`；预注册通过线维持 ≥77.18%），证明本仓适配实现与官方等价后再用于动物域——重实现类论文的标配防线（风险登记册 R4） | ✅ **终判 PASS（2026-09-02 协调者收官收编）**——Phase A ✅（协议核实 + 数据就位，verify PASS 113,156）；Phase B 三流 pretext 全部 300/300 ✅；三流线性评估 joint **74.30%** / bone **71.51%** / motion **67.84%**；3s 融合 **top1=77.97% / top5=95.78%（n=16,487）≥ 预注册线 77.18% → PASS**，与论文正文 78.9% 差 0.93pp 在容差内——本仓适配实现等价性成立（风险登记册 R4 解除；joint 单流 74.30% vs 官方 README 74.34% Δ=−0.04pp 的 PASS_BAND 保真度证据保留在案）。证据：`reports/ntu-phaseB-3s-ensemble.json`（per_stream 元数据完整）。成文段见下方 §4.4 成文块 |

#### §4.4 成文块（英文正文素材，装配目标 latex/sections/04-experiments.tex；数字全部溯源 `reports/ntu-phaseB-3s-ensemble.json`）

**Implementation-equivalence verification on NTU60.** Because AimCLR, SMQ, and the self-training machinery are re-implemented in this repository rather than imported, we verify implementation correctness on the human-domain benchmark before applying the pipeline to animal data. Following the linear-evaluation protocol of AimCLR on NTU60 cross-subject, each stream (joint, bone, motion) is pretrained for 300 epochs and evaluated with a frozen encoder and a fully-connected classifier; the three streams are then fused with the official fixed weights (joint 0.6, bone 0.6, motion 0.4). The re-implementation reaches 74.30% (joint), 71.51% (bone), and 67.84% (motion) top-1, and the three-stream fusion attains **77.97% top-1 (95.78% top-5; n = 16,487)** — exceeding our pre-registered acceptance line of 77.18% and lying within 0.93 pp of the 78.9% reported in the AimCLR paper body. We therefore treat the adapted implementation as equivalent to the official one for the purposes of this study. 【公开基准层 NTU60 xsub；融合权重与协议移植自官方 ensemble_ntu_cs.py，入口 `scripts/ntu_ensemble_3s.py`】

> 成文纪律：①数字零池外（78.9% 口径经 W9 核实；正文 78.9% 与 79.18% released-model 复测之区分已在 §2.2 处理，本段只引 78.9%）；②层级标注【公开基准层】强制随行；③本段不进 Abstract/Intro——R4 是防线证据不是贡献主张，落点 §4.4 即止。

## 5. 消融与分析（Ablation and Analysis）— tab3（W21 映射版）

> 填写纪律（AGENTS.md 硬规则 3/4）：每格标注**层级**与**来源报告路径**；无实验支撑的格子一律 PENDING，禁止用相邻数字充数。

| 开关 | 关闭后果假设 | 对应 claim | 当前证据状态（W21 映射） |
|------|------------|-----------|------------------------|
| − 自监督预训练 | kNN 掉回随机水平附近 | C3 | ✅ **可填并升级为完整梯度叙事（W45 入表，2026-08-26；full 档=W39 执行、低资源梯度三档=同日用户裁决加跑，纯 CPU 让行执行）**——**预训练收益随标注资源增加单调衰减：+21.21 → +9.85 → −2.27 → +0.15 pp @ train=88/176/352/1760（spc=5/10/20/100，两臂 scratch/warm ×3 seeds×50ep 严格等预算，唯一差异变量 = encoder 初始权重）**。① 极低资源端坍缩近随机：spc5 scratch **10.61%±4.29 ≈ 2.3× 随机基线 4.55%** vs warm-init **31.82%±9.82（=7× 随机、3× scratch）**，同 seed 配对 **3胜0平0负**（最小配对差 +13.64pp）→ 极低资源下自监督预训练是性能存续的必要条件；② spc10 warm **82.58%±2.14** vs scratch 72.73%±8.09（+9.85pp，2胜1平），且 warm 方差较 scratch 收窄 3.8×——稳定性收益独立于均值收益；③ spc20 轻微反转 −2.27pp（88.26±2.14 vs 90.53±2.83，幅度 < 两臂 std，位于理论交叉区，如实披露）；④ 饱和档 Δ+0.15pp n.s.（96.97±0.28 vs 96.82±0.37，2胜1平）+ 收敛动力学占优（best_epoch 15.0±4.0 vs 21.7±12.9）。**C3 微调维度直接实证成立**（tab3 原假设「−自监督预训练 → 掉回随机水平附近」在极低资源端兑现），与 kNN 表征可分性证据（P0.1：20.89%=2.51×随机，公开真实层）构成**两条互补证据线**。【合成层 / val_acc / Δ 各档内配对计算；warm 权重来源公开真实层 P0.1——两层口径分列】来源：`reports/ablation-gradient-2026-08-26.md` §1–§4 + `reports/w39-ablation-gradient-spc{5,10,20}-2026-08-26.json` + `reports/w31-ablation-pretrain-2026-08-26.json`。⚠️ 引用纪律（W39 报告 §4 强制随行）：① 低资源档与饱和档必须成对呈现（单引 +21.2pp 即 cherry-pick）；② 梯度档(CPU)与 full 档(GPU)精度列禁止混排单表——趋势论证基于各档内 Δ 不受影响；③ val=22 粒度随行披露（spc5 单样本=4.55pp，+21.2pp 属方向性大效应而非精确效应量）；④ C3 的 scratch-encoder kNN 对照臂仍未检验（本实验无该臂），表征维度原假设闭环仍待补 |
| − 语义 warm-start（换通用 SSL 预训练） | 低标注预算端到端性能下降 | C6 | ⚠️ **R16 协议修正 supersede（见 tab3 表下 R16 修订注）**；原 ✅ **可填（P0.8 落地，2026-09-04）**——同协议方法对比消融：E7/E8 端到端管线逐字固定（seeds→锚点伪标签迭代→seeds+pool 线性头→val，泄漏防护一致），**唯一变量 = backbone 初始化**（PSD 语义 warm Y_CKPT vs AimCLR 通用 SSL@InterPet4D epoch120）。top-1 全预算 PSD warm 占优（下列为 n=3 首轮，论文主数字已切 n=10 口径见判读段）：spc2 **30.95±2.23 vs 27.38±4.45**、spc4 **32.14±2.53 vs 24.41±8.91**、full **33.93 vs 30.36**，两臂均超 scratch（spc2 25.00）；**macro-F1 不一致如实并报**：spc2/full 下 AimCLR 反超（15.21/18.25 vs 12.44/14.65）——warm 特征在极不均衡池上锐化多数类（top-1↑/少数类召回↓）。判读：~~n=3 未达显著~~ → **P1.0 扩容 n=10 后 top-1 全预算显著**（spc2 +5.89pp 10-0-0 p=0.002 / spc4 +7.32pp 7-0-3 p=0.016），macro-F1 spc2 持平（−0.42pp 5-5 p=0.77，n=3 反转消解为种子噪声）、spc4 方向偏 warm 未显著（p=0.084 禁写显著）。机制句="低预算下起作用的不是 SSL 本身，而是与目标标签结构对齐的特征几何——语义 warm-start 恰好供给该对齐"。公平性披露随行：①各臂原生预处理（AimCLR: NTU 视图+序列归一+conf<0.5 置零；PSD: 原始 (30,24,3)）；②AimCLR 预训练源为 mocap，与 in-the-wild AK 存在采集域差，本臂为通用 SSL 迁移下界参照非其最优调参；③scratch 数字引 p07 未重跑。【公开真实层】来源：`reports/p08-aimclr-arm-2026-09-04.md` + `.json`（驱动 `scripts/run_p08_aimclr_arm.py`，复用 p07 协议函数） |
| − 无监督分割 | 无法处理连续流 → 只评片段级 | C4 | ✅ **可填（W38 三臂消融落地，W36 入表）**——分割策略消融（seeds 伪 GT 协议，4 episodes，同 episode 同 GT 同匹配数学）：SMQ 自适应边界 **0.458±0.049** > 等段数均匀滑窗 **0.399±0.035** > 随机切分 null **0.323±0.022**；预注册三门全过（方向 3/4、Δ=0.059>噪声 std 0.0488、uniform ≥ null）→ 结构化分割不可约简为任意切分成立。【公开真实层】来源：`reports/p02-seg-strategy-ablation-2026-08-25.md` §3+§5（复现门与 eC-seeds 定稿逐位一致）。⚠️ 双向论证强制随行披露（W38 §4）：①固定网格旁证臂 0.453±0.027 与 SMQ 统计等效（Δ=0.005<std）——自适应边界对最强平凡基线的边际收益为边缘性，论文措辞以"优于任意切分 + 等段数控制下最优"为准，禁写"大幅领先一切基线"；②boundary F1 上均匀窗反超 SMQ（0.396 vs 0.343），两指标方向不一致禁止单指标引用；③n=4 episodes 样本量有限如实标注。 |
| − 锚点引导 | 聚类漂移、纯度下降 | C5 | ✅ 可填：锚点引导原型聚类纯度 **0.5339** vs 随机分配基线 0.3306（**1.615×**）；种子比例 25%→100% 纯度增益 ≤1pp（原型质量早期饱和）。【公开真实层/P0.3】来源：`reports/p03-jia-phasea-2026-08-24.md` §4、§4.2 |
| − 伪标签迭代（单轮） | 覆盖率停滞 | C5 | ✅ 可填：迭代 r0→r1 池精度 **0.5125→0.6913**（峰值 +17.88pp）；B-1-off 对照行：未校准原型路自迭代 τ\* 退化至 0.0005，迭代无法启动（冻结 0.5000）。【公开真实层/P0.4·物理先验伪标签共识口径】来源：`reports/p04-tcl-2026-08-24.md` §3、§4 |
| − 主动学习（改随机采样） | 达标预算超 200 片段 | C7↓（裁决①探索性发现） | ✅ 可填（**负结果如实**）：合成层短预算档熵采样未显示优势——b=100 随机 +7.9pp（2/3 seeds 同向，seed43 反向）/ b=200 随机 +7.1pp（3/3 seeds 同向）；"达标预算"假设被反向证据覆盖且 C7 经裁决①降级，本行以探索性发现形式呈现。【合成层/W14】来源：`reports/w14-p05-al-efficiency-2026-08-24.md` §3.1 + `reports/p05-al-efficiency-short-2026-08-24.json` |
| **种子噪声注入（v0.3 新增）** | 向规则引擎种子注入 {10%, 20%, 30%} 标签噪声，验证伪标签迭代不放大种子错误——语义层鲁棒性防线（对应 method.md §3.3.2 设计决策） | C5 | ✅ 可填：purity q=0% **0.5339** → 10% 0.5267±.0013 → 20% 0.5201±.0013 → **30% 仅降 3.1pp 至 0.5025±0.0063**（仍为随机基线 1.52×，≥1.5 门保持）。【公开真实层/P0.3】来源：`reports/p03-jia-phasea-2026-08-24.md` §4.3 |

> **R16 修订注（2026-09-05）**：−语义 warm-start 行与 E7/E7b/E9 端到端数字产出于协议错误（最终头消费池真标签+oracle 停止），已被修正协议重跑 supersede：AK v1/v2 低预算臂近随机（v1 warm 9.8±7.5 vs aimclr 15.2±5.1，macro-F1 aimclr 占优 Holm 0.043——warm 全预算占优主张在低预算端反转）；NTU 保留率 90.6% 仍过预注册 90% 线；全监督参照与纯监督对照（EP3、五臂、微调对照）不受影响。证据 `reports/r16-endtoend-pseudo-2026-09-05.json` + `reports/r16-ntu-pseudo-2026-09-05.json`；论文 04/05 已按修正数字装配。

敏感性扫描：种子比例 / 聚类数 K / 置信度阈值 τ / 迭代轮数 / 类别不平衡处理（frequency-aware margin vs 重采样 vs 无处理）/ **样本量梯度 samples_per_class（已落地四档，见下方叙事段）**。
分析段统一采用四步结构（observe→interpret→implicate→next），异常值显式标记不静默剔除。

### 低资源梯度叙事段（v1.1 新增，W45）——标题 *Low-Resource* 的定量注脚
> 本段供正文 §5 分析与 Intro/Abstract 写作窗取用；三层口径与成对呈现纪律随行，取用时不得拆散。

- **一句话主张**：低资源场景下自监督预训练是性能存续的必要条件，其边际价值随标注预算增加而单调衰减至统计等效——**标注资源越稀缺，免标注先验越不可替代**。
- **曲线整条呈现**（合成层，两臂 scratch/warm ×3 seeds×50ep 等预算，Δ=同 seed 配对 warm−scratch）：train=88 → **+21.2pp**（scratch 10.61%≈2.3×随机坍缩近随机带 4.55%，warm 31.82%=7×随机）；train=176 → **+9.9pp** 且 warm 方差较 scratch 收窄 3.8×（可靠性增益独立于均值增益）；train=352 → **−2.3pp** 进入交叉区（幅度 < 两臂 std，如实披露）；train=1760 → **+0.15pp n.s.**。
- **机制解释（一句版）**：样本越少，从数据中学出可分表征越不可能，预训练 encoder 提供的跨域先验成为唯一可依赖的结构；资源充足后监督信号自足，先验边际价值归零甚至轻微为负（跨域特征与目标分布错配成本开始可见——spc20 反转的机制候选）。
- **与 E5 warm-start 叙事呼应（一纵一横）**：E5（语义层 warm-start，82.0%@20 片段）是初始化带来的**绝对水平**证据；本曲线是物理层预训练的**边际价值**证据。两者共同刻画"小预算可用性"，互不替代、不可混报口径。
- **双向论证随行**：正方 = 四档横跨两个数量级训练集规模 + 配对方向一致性（spc5 3/3 全胜，tiny-n 符号检验 p=0.125 已属强方向证据）+ 与迁移学习文献一致的单调衰减形态；反方 = n=3 对 spc10 的 +9.9pp 仅方向性支持、CPU(梯度档)/GPU(full 档) device 混杂禁混排原始精度列、同域预训练对照缺失使「预训练收益上限」不可估。
- 来源：`reports/ablation-gradient-2026-08-26.md`（判读报告含交叉点边界与引用纪律全条款）。

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

- [x] ~~所有 `[PENDING]` 数字待 P0.2-P0.5 报告归档后回填~~ ✅ W36 终填完成（2026-08-25/26 三轮）——可填占位符全部按索引表溯源终填，tab3 六行零 PENDING
- [x] ~~tab3 −自监督预训练行~~ ✅ W39 full 档落地入表（2026-08-26，零结果如实 + 引用纪律随行）；✅ W45 升级四档梯度叙事 + 低资源梯度叙事段（同日，交叉边界前送 Limitations L10——重基于 W46 后撞号重排，原拟 L9）
- [ ] （可选加固项，非阻塞）spc{35,50} 细档定位预训练收益交叉点 / seeds 扩至 5–10 收紧置信区间——纯 CPU 分钟~半小时级，见 `reports/ablation-gradient-2026-08-26.md` §2-Next
- [x] NTU 三流补全链数字落地后：更新 §4.4 实现正确性行（3s 融合 vs 预注册线 ≥77.18% 终判）与 R4 状态 ✅ **2026-09-02 收官收编**——三流全交付：joint 74.30 / bone 71.51 / motion 67.84，3s 融合 **77.97% ≥ 77.18% → PASS**（top5 95.78%, n=16487, alpha 0.6/0.6/0.4）；证据 `reports/ntu-phaseB-3s-ensemble.json`；E6 曾双失败 HALT 已双修复（入口 cwd 对齐 + 时间戳前缀行内解析），JSON 干净归档
- [ ] E4 对比研究"动物域参照行"若工程量超支，砍项决定须上报用户
- [ ] 合成层规模登记待 K9 资产移植（`docs/assets-map.md` 流程）
- [ ] Q3c round2（扩展池增强）结果落地后复核 E4 公开真实列第二行表述

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 窗口骨架：E1-E6 实验-claim 映射 + 三层协议 + 图表规范 |
| v0.2 | 2026-08-23 | 对抗评审加固：统计协议（配对检验+多重校正）+ 折间方差披露 + NTU 验证行【需用户决策】+ 数据集引用义务 |
| v0.3 | 2026-08-23 | 第二轮对抗评审：E6 增 Y′ 来源要求与判据固定条款 + 消融增种子噪声注入行 + 敏感性增不平衡处理开关 |
| v0.4 | 2026-08-24 | W21 诚实修正：清除 E5/E6 过期占位污染（`0.691 ± 0.013` 冒充残留）——E5 负结果如实入册（W14 合成层短预算：熵未占优 + softmax 饱和诊断，C7 标 [PENDING-用户措辞裁决]）；E6 回填 C1 真证据（合成层 small 档：墙钟 7.32×保守 ≥3×、精度 +2.27pp、标注单元打平，full 档待 GPU） |
| v0.5 | 2026-08-24 | W21 tab3 消融表映射：六行中四行完成逐格溯源填充（噪声注入/P0.3 §4.3、锚点引导/P0.3 §4+§4.2、伪标签迭代/P0.4 §3+§4、主动学习负结果/W14 §3.1），−自监督预训练与−无监督分割两行 PENDING 不编造 |
| v0.6 | 2026-08-25 | 用户裁决 C7 选项①落地：E5 降级"探索性发现"（标题/目的句/处置块改裁决记录），效率主张正文禁用，升级通道注明（W23 warm-start 或 full-budget 正证据 + 用户再裁）；tab3 AL 行与 fig4 落点同步 |
| v0.7 | 2026-08-25 | W34 校准轮（relay Q2 full 档回写）：E6 成本主张以 **6.07×** 定稿（保守 ≥3× 由 full 背书，CPU 污染疑虑闭环）、精度措辞校准为**统计等效（full −0.91pp / small +2.27pp，均 <2.3pp）**并删除"略胜"表述、R2 注记标"成本维度可转 ✅ 待终裁"；E2 基线标注勘误（"滑窗基线"→等段数随机切分 null，全仓无滑窗方法臂）；tab3 −无监督分割行排查定案维持 PENDING + 新实验设计入册；对比研究分割策略行同步注记 |
| v0.8 | 2026-08-25 | **W36 终填轮**：种子清单 [PENDING] 终填（逐实验溯源 + 缺口如实披露）；E4 增 tab2 公开真实列第二行（Q3c 44.90%=1.80×随机，4 类部分口径，per-class 分解强制同框 + 类不平衡前送 L7）；§4.4 NTU 行（R4）改事实版状态（pretext 300/300 收官/线性评估基建就绪/执行数未产出/三流范围缺口待用户裁决，禁引未落盘数字）；tab3 两行 PENDING 注记刷新（W38/W39 已注册未开工）；未验证项清单收口重排 |
| v0.9 | 2026-08-26 | **W36 终填轮·二轮增补**（重基 master 后收编 W33/W38 并行成果）：tab3 −无监督分割行 ✅ 入表（W38 三臂：SMQ 0.458±0.049 > 均匀滑窗 0.399±0.035 > null 0.323±0.022，预注册三门全过；网格旁证臂平价 + F1 指标反向双披露随行）；tab3 −自监督预训练行注记更新（W39 执行中）；§4.4 NTU 行 R4 更新（joint 74.30% Δ−0.04pp PASS_BAND 保真实证 / 预注册线 NOT_TRIGGERED 如实归档 / 裁决 A 三流补全链 ETA~49h 待二次收编）；对比研究分割策略行 ⏳→✅ 三臂对照落地 |
| v1.0 | 2026-08-26 | **W36 终填轮·三轮收口**（重基吸收 W39 收编）：tab3 −自监督预训练行 ✅ 入表（W39 full 档零结果如实：Δ=+0.15pp n.s.、配对 2胜1平、收敛动力学占优；预训练价值锚定 kNN 表征证据 + C3 原假设未直接检验的引用纪律随行）——**tab3 六行全部映射溯源完成，消融表零 PENDING**；头部状态与未验证项清单同步收口 |
| v1.1 | 2026-08-26 | **W45 预训练价值梯度轮**：tab3 −自监督预训练行升级完整梯度叙事（spc5 +21.21pp scratch 近随机坍缩 → spc10 +9.85pp 且 warm 方差收窄 3.8× → spc20 −2.27pp 交叉区 → full +0.15pp n.s.——**收益随标注资源单调衰减，C3 微调维度直接实证**，与 kNN 表征证据构成双证据线）；新增「低资源梯度叙事段」（标题 *Low-Resource* 定量注脚，与 E5 warm-start 一纵一横呼应）；敏感性扫描增样本量梯度开关；交叉点诚实边界（spc≈15–20 区间括出、未细档定位）前送 conclusion-limitations.md **L10**（重基于 W46 后撞号重排：W46 已占用 L9=round2 负迁移边界）；outline.md 状态行同步移交协调者/写作窗（本窗无领地） |
| v1.2 | 2026-09-04 | **E7/E8 端到端补强轮**（冲击 PR 的预注册实验，驱动 `scripts/run_p07_endtoend_ak.py`）：新增 §4.3 E7 段（全管线端到端 spc2=2 片段/类 ≈13% 标注预算达全监督 91%：30.95%±2.23 vs 33.93%；warm vs scratch 端到端 +6.0pp、macro-F1 2.8×）+ E8 预算曲线（spc2/spc4/full）+ tab2 端到端行；Abstract/Intro/§4.3 同步回填；outline C6 行旧"池精度而非端到端"保留意见解除 |
| v1.3 | 2026-09-04 | **P0.8 同协议消融 + K9 预注册轮**：tab3 新增 −语义warm-start 行（AimCLR@InterPet4D 替代 PSD warm，端到端协议唯一变量替换——top-1 全预算 warm 占优 +3.6/+7.7/+3.6pp、macro-F1 混合如实并报、n=3 未达显著禁写"显著"、mocap 域差与原生预处理公平性披露随行；`reports/p08-aimclr-arm-2026-09-04.md`）；K9 真实域试点结案为**数据物理不存在**（k9 仓 ADR 0008 v1.7 用户 2026-08-19 已确认 682 片段零标注并废弃该路径；生成脚本已删、归一化约定不可核实→无监督域差探针亦放弃，避免预处理污染）——唯一诚实处置 = 预注册协议 `docs/paper/k9-pilot-preregistration.md`（PSD-K9-PREREG-001：双标注 κ≥0.60 门/session 级切分/单一主终点配对判据/禁再分发），§4.1 装配引用句；DA 声明审稿快照由 commit 哈希改为不可变 tag `review-snapshot`（消除自引用死锁） |
| v1.4 | 2026-09-04 | **P1.0 种子扩容轮**（解除"n=3 未显著"审稿风险）：E7/P0.8 关键对比 n=3→10（驱动 `scripts/run_p10_seedexpansion.py`，特征确定性只重跑协议层，scratch 每 seed 独立随机骨干）——**warm vs aimclr top-1 spc2 +5.89pp 10-0-0 p=0.002 / spc4 +7.32pp p=0.016 全预算显著**；macro-F1 spc2 持平 p=0.77（n=3 反转消解为种子噪声，"多数类锐化权衡"叙事降级）；warm vs scratch +7.32pp/+9.97pp 双显著 p=0.002。**论文主数字切换**: warm@spc2 31.96±2.45=全监督 **94%**（原 91%），Abstract/Intro/§4.3/tab2/tab3/§4.4 六处联动刷新；`reports/p10-seedexpansion-2026-09-04.md` |
| v1.5 | 2026-09-04 | **E7b AK v2 预注册复现轮**（把天花板归因变成检验）：协议 PSD-AKV2-PREREG-001 构建前冻结→多段重提取 352 clips（8 类空间，bark/sit 门失败如实披露）→**EP3 触发 DATA_BOTTLENECK_CONFIRMED（v2 full 37.50% = v1+3.57pp ≥ +3.0pp 冻结线）**；EP2 复现增强：warm spc2 33.23（88.6% 保留@6% 预算）、warm vs aimclr **双指标 10-0-0 p=0.002**（v1 macro-F1 持平判定为单片段标签结构伪影，AimCLR v2 坍缩至 1.71%）；§4.1 数据集句+§4.3 E7b 段+tab2 v2 行+E7 天花板句升级装配；v1 数字零替换；驱动 `scripts/run_p11_ak_v2_build.py` + `run_p12_ak_v2_replicate.py`；`reports/p12-akv2-replication-2026-09-04.md` |
| v1.6 | 2026-09-04 | **P1.3 端到端微调诊断对照轮**（回答『绝对精度为何不刷』）：v2 解冻骨干端到端 full **32.99±3.94 < 冻结头 37.50**（过拟合吞先验）/ spc2 **9.72** vs warm 33.23=3.4× 坍缩——『天花板非冻结所致+解耦实证更优』入 §4.3 E7b 段末（control 措辞非预注册终点）；驱动 `scripts/run_p13_v2_finetune.py`；`reports/p13-v2-finetune-2026-09-04.md` |
| v1.7 | 2026-09-04 | **E9 NTU 低资源保留率轮（重炮命中）**：PSD-NTU-PREREG-001 实验前冻结→冻结 pretext 探针三臂——10%+selftrain **74.11±0.13 = 保留率 99.5%**（参照 74.45，判据 GENERALIZES≥90%）；10% 纯线性 66.05（伪标签迭代 +8.0pp）；对照 TCL 保留率 93.3% 仅比行为不比绝对值；参照臂 74.45 vs 官方 LE 74.30 保真自检过；§4.4 E9 段+tab2 行+Intro Para6 半句装配；驱动 `run_p14_ntu_featuredump/lowres.py`；`reports/p14-ntu-lowres-2026-09-04.md` |
| v2.4 | 2026-09-07 | **R22 审计轮（口径切换封口）**：R22a 逐行一致性 5 发现全修（**MAJOR: 台账缺 E9d 10-seed 块——本行同轮补齐**，3-seed CONFIRMS 翻转史如实入块；74.3/74.45 双口径限定词；表4 E9d 行补 ten seeds；fig5 死加载清理）；R22b 恶意审稿 10 发现（1 CRITICAL+7 MAJOR+2 MINOR）：E9 NTU60 3-seed 不一致→**10-seed 扩容上 GPU**（(a)/(c) 与 09-05 工件逐位一致自检过）；摘要/结论/intro 补 linear-only 88.7% 语境；82.0% 教师账目算全（clean 域 2200 全预算教师+offset 学生 20 边际）；E7c 犬科归因收窄（tier-specific，PanAf 同粒度反例注明）；E9b/c 正式检验入正文（NTU120 显著 t=12.0/UCF101 边缘不显著如实）+工件 r22-e9bc-gap-tests；fig5 caption 位移措辞修正。遗留=用户人工项（外部时间戳 OSF）+E6 第三臂候选实验（GPU 空闲后评估）。 |
| v2.3 | 2026-09-07 | **E9b/E9c 10-seed 终口径轮（种子扩容链收官）**：NTU120/UCF101 b 臂按 E9 系列先例 3→10 seeds 重跑（判据/臂/预算零改动）——**E9b 88.86% PARTIAL**（48.29±0.76，+2.9pp，10/10 种子高于线性；3-seed 初读 89.24% 留档）/ **E9c 66.64% FAILS**（15.40±1.89，+1.4pp，2/10 种子低于线性不做同向声明；3-seed 62.62% 方向未翻转）；两协议 dated Amendment（ntu120 A3 / ucf101 A2）；正文 5 处+fig5 数据源切换 09-07 JSON 重绘；看门狗链全终（yolo DONE/PanAf EXTRACTED/P7_DONE），预注册实验队列清空。证据 `reports/p5b-{ntu120,ucf101}-retention-2026-09-07.json`。 |
| v2.2 | 2026-09-07 | **P7/P2′ 双线收官轮（ADR 0009 增强线）**：**E9d PanAf500 = 首个动物域公共基准保留率点，retention 93.33% → 预注册 CONFIRMS**（PSD-PANAF-PREREG-001 跑前冻结 a73cc70；(c) 60% / (a) 52% / (b) 60/56/52 = 93.33%，池 357-358；链式门控全自动产出：ape 微调 100ep→PanAf500 推理 500 视频 det 0.821→三臂）；跨域四点梯度 NTU60 90.6→NTU120 89.2→PanAf 93.3→UCF101 62.6，动物域首次 CONFIRMS 级证据。**P2′ 判据 LABEL_BOTTLENECK**（PSD-SA-PREREG-001：YOLO11x-pose(dog-pose 100ep, mAP50 0.834) 重提取 197 视频→全监督 30.36% vs 33.93% = −3.57pp < +3pp 线）——更强提取器未抬全监督天花板，标签错位瓶颈结论再获受控确认；低预算臂按协议跳过。工程事故如实：ep28 dataloader 死锁（workers=0 恢复提速 8×）；p18 假结果未遂（keypoint_conf 非法参数→全视频静默失败→pkl=旧数据复制，full_ref 与基线逐位一致触发人工核查）——"匹配不全熔断"防再犯。 |
| v2.1 | 2026-09-05 | **P5 跨域双点轮（ADR 0009）**：E9b NTU120（HRNet 2D）retention **89.24% → PARTIAL**（PSD 臂 48.50% vs 10% 线性 45.42%，+3.1pp，3/3 种子）；E9c UCF101（独立第三域）**62.62% → FAILS**（PSD 臂 +0.4pp 仅方向一致）——跨域梯度与参照臂质量单调对应，pretext 特征质量=保留率主导因素，与 R16 犬科负边界同构入限制叙事。学术诚信注：首轮 b 臂 str/int 评分错配伪记 0.0/FAILS，经"精确零不可能"检查定位修复后重跑（臂/预算/种子/判据零改动），修复先于读数，两协议 Amendment 落档；pretext 协议偏离（ST-GCN→joint-level MLP 适配任意关节数）入 Amendment 2。驱动 `run_p5b_generic_retention.py`（NTU120 679s / UCF101 850s）。 |
| v2.0 | 2026-09-05 | **R17 回归审计轮**：①macro-F1 评估器类索引 bug 发现并修正传播（EP3"反向"与微调对照"反向"两条披露撤回，修正后双指标同向）；②Holm 族口径统一为逐层（m=6，工件重算：v1 mf1 spc2 0.023/spc4 0.049、v2 scratch-mf1 0.012）；③E7b v2 种子数如实 14/256≈5.5%（sit 无训练段）；④chance 口径调和（名义 8.33→12.5 vs 实现 11.1）；⑤E9 补 linear-only 88.7% 对照语境；⑥共识门惰性披露补入 E7/E9；⑦五臂 macro-F1 12-slot 口径披露；⑧p07 勘误文件+评估器 bug 注记。 |
| v1.9 | 2026-09-05 | **R16 端到端协议诚信修正轮（学术诚信级）**：对抗审稿实锤 E7/E7b/E9/P0.8/P1.0 最终头消费池真标签+precision-drop 停止消费 oracle——"94%@13%/99.5%@10%/88.6%@6%"与实现不符；修正协议（种子真标签∪池伪标签头；无 oracle 停止）重跑：AK v1/v2 近随机（负边界入文），NTU 90.6% 仍过预注册 90% 线（GENERALIZES 存活）；head_calib GT 无关诊断排除门控尺度解释；E4 首末配对 p 补 Holm（校正后 0.090，工件 r16-holm-p04）；E1 ±4.45→±4.04 与 tab3 AL 2/3 seeds 同步；两预注册协议追加 dated §7/§8；p08/p10/p12/p14/w14 报告追加勘误；fig5 重绘（tier-dependent 叙事）。 |
| v1.8 | 2026-09-04 | **R8 对抗审稿+引用实查修复轮（学术诚信级）**：①TCL 82.7%/88.6%/NTU60 经原文 PDF 全文检索证伪（NTU 0 命中），系池内继承假引用——正文 E4/E9/Intro 全链删除，bib 作者修正（Ankit Singh/Abir Das），outline/method/related-work/skeleton 同步清污；②E7b 类别空间混淆修复：8 类同空间对照 v1(8cls)=25.96% → 同空间差 +11.54pp（数据瓶颈归因修复后于 top-1 成立，macro-F1 反向双报）；③E9 修复：30× 不可解释数字删除、8.0→8.1pp、ceiling→reference、单子集/适配/协议依赖披露补全、TCL 对比删除；④Y_CKPT 源视频泄漏披露补全+independent→pre-registered parallel；⑤微调句 macro-F1 反向双报+ superiority 限定双指标一致区；⑥tab3 (10 seeds) full 列限定、both-arms-beat-scratch 限定 spc2、only-backbone→+预处理、artifact→hypothesis、geometry 断言降格、Abstract 94% 补绝对锚点、Intro ten-seeds 修饰语修正、13% 口径澄清；⑦AimCLR++ 77.2→80.9（官方仓库证伪）、79.18 归属措辞、NTU60 补引 Shahroudy2016、aimclrpp 题名按 CrossRef；⑧两预注册协议追加 dated 修订节（非静默改）。审稿 agent 判定 Major Revision→修复后证据-主张对齐恢复 |
