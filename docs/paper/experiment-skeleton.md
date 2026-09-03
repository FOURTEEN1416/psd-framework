# 4-5. Experiments & Ablation 骨架（三层口径 · P0.6）

> Owner: `docs/paper/experiment-skeleton.md` · W5 窗口 2026-08-23 · 状态: **终稿候选 v1.2（2026-09-02 协调者三流收官收编轮；v1.1 = W45 预训练价值梯度轮）**——E1-E4 数字在档；E5 负结果如实入册且 C7 换轨定稿；E6 成本 6.07× 背书 ≥3×；tab3 六行全部映射溯源完成；**§4.4 NTU 行 R4 ✅ 终判 PASS（2026-09-02）：三流 pretext 300/300 ×3 + LE joint 74.30 / bone 71.51 / motion 67.84 + 3s 融合 77.97% ≥ 预注册线 77.18%（top5 95.78%, n=16487）——本仓适配实现与官方等价性成立，风险登记册 R4 解除**；v1.1 增补（W45）：tab3 梯度叙事 + 低资源梯度段 + L10 前送（保持不变）
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
- **已可填**：kNN top-1 **20.89% ± 4.45% (公开真实层/P0.1)** vs 随机 8.33%，倍率 **2.51×**（fold 明细：20.00/15.56/26.67/22.22/20.00；⚠️ 折间极差 11.1 个百分点，成稿须披露并归因于 225 序列的小样本方差）。来源：`reports/p01-knn-result.json`。
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
- 人类域参照行（明确标注为他域参照，不计入本文结论）：TCL CVPR 2021 82.7% @10% vs 全监督 88.6%。

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
| − 无监督分割 | 无法处理连续流 → 只评片段级 | C4 | ✅ **可填（W38 三臂消融落地，W36 入表）**——分割策略消融（seeds 伪 GT 协议，4 episodes，同 episode 同 GT 同匹配数学）：SMQ 自适应边界 **0.458±0.049** > 等段数均匀滑窗 **0.399±0.035** > 随机切分 null **0.323±0.022**；预注册三门全过（方向 3/4、Δ=0.059>噪声 std 0.0488、uniform ≥ null）→ 结构化分割不可约简为任意切分成立。【公开真实层】来源：`reports/p02-seg-strategy-ablation-2026-08-25.md` §3+§5（复现门与 eC-seeds 定稿逐位一致）。⚠️ 双向论证强制随行披露（W38 §4）：①固定网格旁证臂 0.453±0.027 与 SMQ 统计等效（Δ=0.005<std）——自适应边界对最强平凡基线的边际收益为边缘性，论文措辞以"优于任意切分 + 等段数控制下最优"为准，禁写"大幅领先一切基线"；②boundary F1 上均匀窗反超 SMQ（0.396 vs 0.343），两指标方向不一致禁止单指标引用；③n=4 episodes 样本量有限如实标注。 |
| − 锚点引导 | 聚类漂移、纯度下降 | C5 | ✅ 可填：锚点引导原型聚类纯度 **0.5339** vs 随机分配基线 0.3306（**1.615×**）；种子比例 25%→100% 纯度增益 ≤1pp（原型质量早期饱和）。【公开真实层/P0.3】来源：`reports/p03-jia-phasea-2026-08-24.md` §4、§4.2 |
| − 伪标签迭代（单轮） | 覆盖率停滞 | C5 | ✅ 可填：迭代 r0→r1 池精度 **0.5125→0.6913**（峰值 +17.88pp）；B-1-off 对照行：未校准原型路自迭代 τ\* 退化至 0.0005，迭代无法启动（冻结 0.5000）。【公开真实层/P0.4·物理先验伪标签共识口径】来源：`reports/p04-tcl-2026-08-24.md` §3、§4 |
| − 主动学习（改随机采样） | 达标预算超 200 片段 | C7↓（裁决①探索性发现） | ✅ 可填（**负结果如实**）：合成层短预算档熵采样未显示优势——b=100 随机 +7.9pp / b=200 随机 +7.1pp（3/3 seeds 同向）；"达标预算"假设被反向证据覆盖且 C7 经裁决①降级，本行以探索性发现形式呈现。【合成层/W14】来源：`reports/w14-p05-al-efficiency-2026-08-24.md` §3.1 + `reports/p05-al-efficiency-short-2026-08-24.json` |
| **种子噪声注入（v0.3 新增）** | 向规则引擎种子注入 {10%, 20%, 30%} 标签噪声，验证伪标签迭代不放大种子错误——语义层鲁棒性防线（对应 method.md §3.3.2 设计决策） | C5 | ✅ 可填：purity q=0% **0.5339** → 10% 0.5267±.0013 → 20% 0.5201±.0013 → **30% 仅降 3.1pp 至 0.5025±0.0063**（仍为随机基线 1.52×，≥1.5 门保持）。【公开真实层/P0.3】来源：`reports/p03-jia-phasea-2026-08-24.md` §4.3 |

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
