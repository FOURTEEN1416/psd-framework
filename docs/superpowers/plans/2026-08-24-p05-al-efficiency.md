# P0.5 主动学习效率实验（W14 窗口）实施计划

> **For agentic workers:** 本计划由 W14 窗口会话内联执行（executing-plans 模式），步骤用 `- [ ]` 追踪。

**Goal:** 产出熵不确定性采样 vs 随机采样的 22 类 val_acc 效率曲线（预算 {20,50,100,200} 片段 × 3 seeds，mean±std 误差棒），JSON 归档 reports/；GPU 占用期间先以 50 epoch 短预算跑通管线，full-budget config 待一键复跑。

**Architecture:** 两层结构——① `psd/training/active_learning.py` 纯函数式采样器 + 增量式 AL 模拟运行器（复用 `STGCNBCTrainer` 冷启动重训）；② `scripts/run_p05_al_efficiency.py` CLI 编排（smoke/short/full 三档 + 断点续跑）。真实池打分作为独立模块函数消费 P0.4 移交池与 `runs/p05_stgcn_bc_full/best.pt`。

**Tech Stack:** PyTorch 2.11.0+cu128（本窗 CPU 执行）、numpy、yaml、pytest。

## Global Constraints

- 禁触：`docs/paper/**`、`dev-docs/decisions/**`、`*ntu*` 文件（用户指令）
- 三层指标口径铁律：本实验主曲线 = **合成层**；真实池打分 = 公开真实层（仅排序清单，禁止冒充 acc）
- TDD 先行：先写失败测试再实现（superpowers:test-driven-development）
- Conventional Commits 中文描述
- 无新鲜验证，无完成声明：所有结论须当次运行证据
- GPU 被 NTU 长训占用（7624/8151 MiB 实测）：短预算全程 CPU
- P0.4 池实况：191 条 < 预算上限 200，如实登记不得虚报

## 关键设计决策（已定，依据见对话记录）

1. **采样策略 = softmax 熵**（单次确定性前向；MC-dropout 成本不可接受且 STGCNBC dropout=0；边际忽略尾部类概率质量；Settles 2009 标准基线）
2. **配对增量协议**：同 seed 下两臂共享随机初始核 b=20；增量 {+30,+50,+100} 由各臂策略以上一累计预算训得的模型打分选择；每预算点**冷启动重训**（固定初始化种子）消除累积混淆
3. **数据隔离**：池 seed=20261 (spc=10 → 220 条)；验证集 seed=20262 (spc=15 → 330 条固定 GT)。均避开 W12 已用 seed42 数据防记忆泄漏
4. **真实池打分**：jsonl → K9 smal_npy 切片 [start,end) → framewise 归一 → 重采样 T=30 → best.pt CPU 前向 → 熵统计 + Top-K 清单；披露为合成域迁移代理排序

---

### Task 1: 采样器 TDD（RED）

**Files:**
- Test: `psd/training/tests/test_active_learning.py`

- [ ] 测试1 `test_entropy_scores_uniform_max_onehot_zero`：均匀分布熵=log(22)，one-hot 熵=0，形状=(B,)
- [ ] 测试2 `test_random_selector_deterministic_excludes_labeled`：同 rng 可复现；候选集不含已标注
- [ ] 测试3 `test_entropy_selector_picks_argmax_excludes_labeled`
- [ ] 运行确认 FAIL（ModuleNotFoundError）

### Task 2: 采样器实现（GREEN）

**Files:**
- Create: `psd/training/active_learning.py`（仅采样器部分）

```python
def entropy_scores(probs: "np.ndarray (B,C)") -> np.ndarray  # -(p*log(p+eps)).sum(-1)
class RandomSelector: .select(pool_size, k, rng) -> list[int]
class EntropySelector: .select(scores, labeled_mask, k) -> list[int]
```

- [ ] pytest 转 PASS → commit `feat(p05): AL 熵/随机采样器`

### Task 3: 模拟运行器 TDD + 实现

**Files:**
- Test: 同文件追加
- Modify: `psd/training/active_learning.py`

- [ ] 测试4 `test_runner_incremental_rounds_tiny`：tiny 模型(base_channels=8,num_stages=2)+tiny 数据(池24/val22/epochs=2/budgets=[4,8])，断言：返回 4 个预算点指标；labeled 并集 ⊆ 池且互斥于 val；两 budget 点集合为嵌套包含
- [ ] 测试5 `test_paired_initial_core_identical_across_strategies`：同 seed 两臂初始 20 集合逐 id 相等
- [ ] 测试6 `test_train_stage_small_batch_no_empty_loader`：n_train < batch_size 时正常训练返回有限 acc（drop_last 规避）
- [ ] 实现 `ALSimulationRunner`：`run_trajectory(strategy, seed)` → `{budget: best_val_acc}`；内部 `_train_stage(samples, init_seed, cfg)` 用 `STGCNBCTrainer`，batch_size=min(cfg_bs, n_train)
- [ ] PASS → commit `feat(p05): AL 增量模拟运行器`

### Task 4: 真实池打分 TDD + 实现

**Files:**
- Test: 同文件追加（fake npz fixture，不依赖 K9 路径）
- Modify: `psd/training/active_learning.py`

- [ ] 测试7 `test_slice_and_resize_to_T30`：fake clip (1041,24,3) 切 [175,196) → resize 后 (30,24,3)
- [ ] 测试8 `test_score_pool_output_schema`：fake 池条目 + tiny 模型 → 返回含 entropy_stats/topk_ids 字段
- [ ] 实现 `score_real_pool(pool_jsonl, clip_loader, model, budgets)`；clip_loader 注入便于测试
- [ ] PASS → commit `feat(p05): P0.4 池熵打分`

### Task 5: CLI 脚本 + 双配置

**Files:**
- Create: `scripts/run_p05_al_efficiency.py`（--mode smoke|short|full，--resume 断点续跑，状态落 runs/p05_al_efficiency/state_*.json）
- Create: `configs/p05_al_short.yaml`（epochs=50，device=cpu）/ `configs/p05_al_full.yaml`（epochs=120，patience=25，device=auto）

- [ ] smoke 模式端到端冒烟（tiny 参数，2 分钟内）
- [ ] commit `feat(p05): AL 效率实验入口与双档配置`

### Task 6: 短预算全量扫描执行

- [ ] 后台启动 short 模式（6 轨迹 × 4 预算点 ≈ 2.2h CPU），监控日志
- [ ] 真实池打分运行归档（Top-{20,50,100,191} 清单 + 分布统计）

### Task 7: 结果汇总归档

- [ ] 汇总曲线 JSON（per-seed + mean±std 误差棒）→ `reports/p05-al-efficiency-short-2026-08-24.json`
- [ ] 报告 `reports/w14-p05-al-efficiency-2026-08-24.md`（四步结构 observe→interpret→implicate→next；双向论证含反方质疑）
- [ ] commit `feat(p05): W14 主动学习效率曲线短预算结果归档`

### Task 8: truth 回写收尾

- [ ] HANDOVER.md §8 新增 W14 行 + §10 记录 + 修订历史 v1.8（行级替换）
- [ ] dev-docs/stage-plan.md 头部状态行更新（行级替换，编辑前重读）
- [ ] commit `docs(handover): W14 窗口交接回写`

## Self-Review

- 规格覆盖：熵选型理由✓ / 预算×seeds✓ / 曲线 JSON+误差棒✓ / 50ep 短预算✓ / full config 一键✓ / TDD✓ / 中文提交✓ / 禁触清单✓
- 占位符扫描：无 TBD
- 类型一致性：select() 返回 list[int]、run_trajectory 返回 dict[float] 各任务一致
