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

## 修订历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-25 | 三窗规划立项 |
