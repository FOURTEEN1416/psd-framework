# NEXT BATCH — W30/W31/W32 任务规划（数据会师 + tab3 补残 + 终稿预备）

> Owner: dev-docs/handovers/NEXT-BATCH-plan.md
> 立项: 2026-08-25 午后，协调者规划；三窗纯 CPU 或排队 GPU，与 relay 队列零冲突

## W30 统一真实扩展池组装（含 APTv2 拓扑映射）

**目标**: 把五路战役产物装配为可训练的 unified real-expansion pool，消除"胶水工作"债务。

**范围**:
1. APTv2 17kp→24kp 拓扑映射（17 四足通用点 → K9Graph 24 点；映射表文档化+语义理由；无法对应的点诚实标注）
2. 15 帧微序列的时序处理策略定案（滑窗拼接升采样 / 直接短序列训练支持 / 仅作预训练池——三选一须论证）
3. 组装器 `scripts/build_unified_pool.py`: 汇聚 {AK partialclass4(4类有标), APTv2 轨迹(规则种子打标), DogSet(运动学先验池), dogpose 静态(增广池)} → 统一 pkl + 溯源 manifest（每样本记 source_channel）
4. TDD: 拓扑映射 spot-check + 池分布统计 + split 完整性
5. 产出: 池文件 + 组装报告（各类样本量/来源构成/与 22 类体系的覆盖关系）

**领地**: scripts/build_unified_pool.py、psd/data/tests/test_unified_pool.py、configs/unified_pool_*、reports/unified-pool-*
**禁触**: docs/paper/**、decisions、*ntu*、relay 文件、各源原始目录（只读）

## W31 tab3 补残之一：−自监督预训练消融

**目标**: 关闭 tab3 第一行 PENDING——同数据同预算下 scratch vs warm-init 对照。

**设计**: 合成 2200 样本（W12 口径）× 两臂 {随机初始化, 加载 P0.1 AimCLR 预训练 backbone} × 3 seeds，50 epoch 短预算即可判方向（C3 的 kNN 已证表征可分，微调差距预期显著）。复用 STGCNBCTrainer。
**产出**: configs/ablation_pretrain.yaml + scripts/run_ablation_pretrain.py + 报告 JSON/md
**GPU 纪律**: 排队位于 relay 队列之后——本窗只交付脚本+TDD+CPU 冒烟；full 运行列队（协调者统一调度或用户手动触发）
**TDD**: 初始化两臂加载断言 + 对照公平性（同 seed 同切分）

## W32 论文终稿回填预备（机械化降耗）

**目标**: 让最终回填窗口退化为"机械替换"——所有非 Q3 依赖的准备工作现在做完。

**范围**:
1. 生成全文数字索引表：扫描 quickref + 各 reports，产出 `{占位符 → 数值 → 来源文件+字段}` 三列机器可读清单（scripts/gen_number_index.py + reports/number-index-<日期>.json）
2. [RESULT-3] 候选评定：按 outline §4 候选（SMQ 1.53× / 伪标签 +17.9pp / C1 ≥3×）逐一给出推荐排序与理由，交用户一句话定夺
3. Introduction/Abstract 中与 C7 相关句式的换轨预改写（warm-start 叙事版本草稿，标记 DRAFT 待终稿窗口合并）
4. 引用三条作者列表级待补的最后尝试（GitHub-Only 渠道）

**领地**: docs/paper/introduction.md（仅 C7 相关句段）、scripts/gen_number_index.py、reports/number-index-*；experiment-skeleton/outline 本轮不动（避免与既有验收态冲突）
**禁触**: figures/**、decisions、*ntu*

## 执行顺序建议

W30 与 W32 立即并行（纯 CPU）；W31 先交脚本冒烟，full 档等 relay 清空。

## W33 NTU60 线性评估复现数（用户已开窗）

checkpoint 就绪（epoch300_model.pt），预注册线 ≥77.18%；GPU 排 relay ALL_DONE 后。

## W34 C1 full 档报告回写 + E6 措辞校准（CPU 立即可跑）

**背景**: relay Q2 已完成（`f2fc789`）——full 档墙钟比 6.07×（成本结论稳），但精度差 **−0.91pp**（small 档 +2.27pp）。按预注册"以 full 为准"，需诚实校准。

**任务**:
1. 更新 `reports/c1-decouple-cost-2026-08-24.md`：增补 full 档章节（两臂全表/方向一致性/精度措辞校准为"统计等效（full −0.91pp，small +2.27pp，均 <2.3pp）"）
2. 同步 `docs/paper/experiment-skeleton.md` E6 节：成本主张保留（≥3× 由 full 背书），精度表述改"无显著精度代价"
3. R2 状态复核：成本维度 full 档确认 → 🟡 可转 ✅（在 skeleton 风险注记处更新）
4. 自助收编（-Permit docs/paper/experiment-skeleton.md）

## W35 W25 片段提点接力 + 规则种子草稿（数据飞轮第二圈）

**触发条件**: relay state.json 出现 Q3a_yolo_dogpose status=done（权重就绪）
**任务**:
1. 用 Q3a 权重对 runs/data_campaign/video/ 759 片段跑提点（复用 ak_pose_extract 管线，DEAD_JOINTS 硬掩码自动生效）
2. 规则种子两段式第二段：harvest_rule_seeds.py 全量跑 → 七类种子草稿池
3. 产出: 片段骨架序列池 + 种子草稿报告（各类分布/置信度/与 AK 域对比）
4. 汇入 W30 统一池（重跑组装器）
**领地**: scripts/harvest_*、runs/data_campaign/video/keypoints_*、reports/harvest-*

## W36 论文终稿回填（条件窗口）

**触发条件**: Q3c 结果 JSON + W33 线性评估数 双双落地
**任务**: experiment-skeleton 全部占位符终填 + Introduction/Abstract DRAFT 合并定稿 + 全文数字对账（用 W32 索引表） + tab3 收尾核查
**前置**: 等 W34/W35 不阻塞——tab2 公开真实列与 R4 数字是仅剩硬依赖

## W37 tab3 末行排查：−无监督分割

**疑点**: P0.2 报告已含"SMQ vs 滑窗基线"对照（0.458 vs 基线）——tab3 该行可能**已有数据可填**而非需新实验。任务：核查 P0.2 报告对照表是否构成"分割策略消融"证据，可填则填（W34 顺路），不可用则给出所需新实验设计。

## W38 tab3 末行实验：均匀滑窗第三臂（CPU 立即可跑，W34 设计落地）

**背景**: W34 排查定案——P0.2 对照系"等段数随机切分 null（蒙特卡洛）"而非滑窗方法臂，全仓无滑窗臂，tab3 −无监督分割行维持 PENDING；W34 已入册最小新实验设计。
**任务**:
1. 按设计实现均匀滑窗分割臂（复用 eval_smq_segmentation.py 评估协议，seeds 规范对齐）
2. 三臂对照: SMQ 运动词量化 vs 均匀滑窗 vs 随机切分 null——同 episode 同协议
3. 产出: reports/p02-seg-strategy-ablation-<日期>.json/md + tab3 该行回填素材（交协调者或 W36 一并入表）
**领地**: scripts/seg_strategy_ablation.py(新)、configs/seg_ablation_*、reports/p02-seg-strategy-*
**禁触**: 既有 p02 报告与 SMQ 代码本体（只读复用）

## W39 W31 full 档执行：−自监督预训练消融（GPU 排 W33 线性评估之后）

**背景**: W31 已交付脚本+TDD+CPU 冒烟（strict 加载打通）；本窗只负责 full 档执行与判读。
**任务**:
1. 监控 W33 线性评估完成（reports/ntu-phaseB-lineareval-*.json 出现）后占卡
2. 跑 scripts/run_ablation_pretrain.py full 档（两臂×3 seeds×完整预算）
3. 判读回填: tab3 −自监督预训练行素材 + 报告 reports/ablation-pretrain-<日期>.md
**领地**: runs/ablation_pretrain_*、reports/ablation-pretrain-*
**禁触**: W31 脚本本体（只读执行）

## W40 数据飞轮效力验证：扩展池微调第二轮（GPU 排 W39 之后）

**背景**: 数据五路战役的终极追问——飞轮真的转得动吗？44.90%（仅 AK 172 片段）在扩展池加持下能否提升？这是"数据飞轮持续供数"主张的首个直接实验证据。
**任务**:
1. 基于 W30/W35 统一池（9844 条五源）构造增强训练集：APTv2 503 轨迹（几何/预训练用途，deferred 标签不进监督）+ DogSet 运动学先验按 W30 判例入对应用途槽位
2. 两轮对照: round1（现 44.90% 配置原样复跑）vs round2（+扩展池增强），同 seed 同协议
3. 产出: reports/p05-public-real-round2-<日期>.json/md——提升则飞轮主张获直接实证；不提升则如实记录并分析域差瓶颈
**领地**: configs/public_real_round2_*、runs/public_real_round2_*、reports/p05-public-real-round2-*
**禁触**: 统一池组装器（只读消费）、AK 原始 pkl

## W36 论文终稿回填（触发窗：W33 数字落地后开）

**触发**: reports/ntu-phaseB-lineareval-*.json 出现即可开（预计数小时内）
**任务**: experiment-skeleton 全部剩余占位符终填（R4 数字/tab3 W38-W39 素材择时）+ Introduction/Abstract DRAFT 合并定稿 + 全文数字对账（用 W32 索引表） + Limitations 终稿（含 44.90% 类不平衡/20-24 有效监督/AL 负结果三件套）
**领地**: docs/paper/**（全开）

## 第二线：数据问题内部攻坚（不等外部邮件）

### W41 像素域规则种子适配专项（W30 移交项，CPU，直攻数据问题）

**背景**: W40 判明扩展池标签断链根因 = rule_seeds.py 依赖 3D z 高度轴+SMAL 索引组，而 APTv2/w35 为 2D 像素域 (x,y,vis)。W40 报告"飞轮下一圈燃料"第 2 条：经 y 翻转+体尺度归一+阈值再校准可产出弱监督信号，**绕开人工预算**。
**任务**:
1. 2D 像素域规则族设计：y 翻转不变性验证 → 体尺度归一（bounding-box 归一替代 z 轴）→ 七类规则阈值再校准（用 AK partialclass4 有标 172 片段做校准集）
2. TDD + 产出 psd/data/rule_seeds_2d.py + 弱标签池
3. 效果门禁: 弱标签 vs AK 172 有标真值的一致率报告（≥ 某阈值才入监督管线，否则如实降级）
**产出**: psd/data/rule_seeds_2d.py + tests + reports/rule-seeds-2d-<日期>.md/json

### W42 标签入环 AL 选样管线（CPU，为 K9 视频 Day-1 备弹）

**背景**: W40 报告"飞轮下一圈燃料"第 1 条——W14/W23 AL 管线现成，缺的是"在扩展池候选上跑选样"这一步。
**任务**:
1. 对统一池 9844 条跑 warm-start 不确定性选样（熵+边际双指标，W23 协议）
2. 产出分级核验清单: Top-100/200 人工核验列表（片段 ID+起止帧+预测标签+置信度）
3. K9 视频到位后 Day-1 即可按清单核验——把"等数据"变成"数据到即冲刺"
**产出**: reports/al-selection-pool-<日期>.json/md + 核验清单文件

### W43 多域适应 round3：修 round2 的稀释病（GPU 排三流链后）

**背景**: round2 −4.08pp 的根因假设 = 多源直接混合稀释（W40 组件消融见类间换位结构信号）。
**任务**: per-source BN 分支或域加权采样两方案择一实现 → round3 对照（round1/round2/round3 三臂）→ 判定扩展池在域适应下是否转正
**产出**: reports/p05-public-real-round3-<日期>.json/md

## 第三线：产品线升级（k9-training-system，野心产品半边启动）

### W-P1 研究成果→产品线移植审计（在 k9 仓执行，遵循其宪法）

**背景**: 研究线已产出产品 Phase 4.1（基座系统产品化）所需的全部核心资产——解耦架构（C1 6-7× 标准演化成本证据=产品 YAML 评分卡演化的业务实证）、22 类管线（97.27%）、数据飞轮全套、warm-start AL（82%@20+梯度曲线）。产品仓 Phase 4 并行实施中（LLM 解释器✅/Mamba✅/编译✅），4.1 待启动。
**任务**（审计先行，零代码改动）:
1. 通读 k9-training-system 宪法/stage-plan Phase 4.1 节/现有架构
2. 产出移植审计报告（放 k9 仓 dev-docs/）: 研究资产清单 × 产品需求矩阵 × 移植方式（assets-map 显式移植纪律）× 依赖冲突 × 工作量预估
3. 跨仓指针回写 psd-framework docs/assets-map.md（反向登记）
**领地**: k9 仓 dev-docs/（审计报告）+ psd 仓 docs/assets-map.md（反向指针）；两仓代码零改动
**注意**: k9 仓有自己的 AGENTS.md 宪法，开工先读其宪法与 HANDOVER

## 修订历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-25 | 三窗规划立项 |
| v1.1 | 2026-08-25 晚 | 增补 W33-W37：NTU 线性评估/C1 full 回写/片段提点接力/终稿回填/tab3 末行排查 |
| v1.2 | 2026-08-25 深夜 | 增补 W38-W40 三窗（tab3 滑窗臂/预训练消融执行/飞轮效力验证）+ W36 触发条件更新 |
| v1.3 | 2026-08-26 深夜 | 用户三连问战略修正：第二线数据内部攻坚三窗（W41 像素域规则种子/W42 AL 选样/W43 多域适应 round3）+ 第三线产品线启动（W-P1 移植审计，k9 仓执行） |
