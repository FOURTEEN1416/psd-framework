# P0.5-AL Warm-Start 协议实验（W23 窗口）实施计划（预注册）

> **预注册声明**: 本文档于实现代码编写**之前**落盘并提交（任务书 §1 Step 2 门禁）。
> 落盘后禁止修改协议字段；若执行中确需变更，须停手留证上报用户裁决后另立版本。
> 用户裁决依据: A2（warm-start + 加噪偏移），2026-08-24，`dev-docs/handovers/W23-p05-al-warmstart.md`

**Goal:** 验证「熵不确定性采样 vs 随机采样」在 warm-start 设定下的效率差异，产出 22 类
val_acc 效率曲线（预算 {20,50,100,200} 片段 × 3 seeds，mean±std 误差棒）——E5/fig4 的正证据候选；
负结果同样如实归档，禁止择优叙事。

## Global Constraints

- 口径: 全程**合成层**（AGENTS.md 三层铁律）；公开真实层路线归 W20，禁止混报
- 禁触: `docs/paper/**`、`dev-docs/decisions/**`、`*ntu*`、`external/**`
- TDD 先行；Conventional Commits 中文；无新鲜验证无完成声明
- GPU 被 NTU 长训占用: short 扫描全程 CPU；full-budget 只备配置+排队，不在本窗运行

---

## 1. Step 1 诊断结论

> 归档路径: `reports/p05-al-efficiency-warmstart-diagnosis.json`（commit `cdd5ffa`，当次运行证据）

best.pt（epoch=38, 训练域 val_acc=96.6%）对三档加噪偏移数据零微调评估:

| noise_std | val_acc | margin(val) | margin(pool) | 带内[40%,70%] | 未饱和(<10) | 判定 |
|-----------|---------|-------------|--------------|---------------|-------------|------|
| 0.10 | **0.4121** | 3.31 | 3.37 | ✅ | ✅/✅ | **eligible** |
| 0.15 | 0.0848 | 12.94 | 12.82 | ❌ | ❌/❌ | 淘汰 |
| 0.20 | 0.0455 | 27.48 | 27.44 | ❌ | ❌/❌ | 淘汰 |

- **选定噪声档: noise_std = 0.10**（唯一同时满足选档判据的档位；熔断未触发）
- 附带现象登记: noise≥0.15 时模型呈「自信地错」退化——acc 塌缩至随机水平而 logit 边际反增
  （12.9→27.5）。域偏移把校准往错误方向推，与 W14 真实池饱和（margin≈100.9）同一族病理，
  支持「先注入域内标注再校准打分器」的 warm-start 动机。
- 数据指纹: 池 seed=20263 / 验证 seed=20264（避开 W14 的 20261/20262 与 W12 seed42）；
  spc 与 W14 相同（池 10/类=220、验证 15/类=330）、T=30。

## 2. 协议（预注册，未事后修改）

### 2.1 与 W14 冷启动协议的差异（仅两处）

| 维度 | W14 冷启动 | W23 warm-start |
|------|-----------|----------------|
| 每预算点初始化 | 随机初始化（固定种子冷启动重训） | **从 best.pt 加载权重微调** |
| 数据分布 | noise_std=0.05, seed 20261/20262 | **noise_std=0.10（加噪偏移）**, seed 20263/20264 |

其余全部保持不变: budgets {20,50,100,200}、seeds {42,43,44}、配对增量设计（同 seed 两臂共享
随机初始核 b=20）、熵打分器选型（softmax 熵，理由见 W14 预注册 §关键设计决策）、
TrainConfig 超参（lr=1e-3 / wd=1e-4 / batch=32 / warmup=5 / cosine）、epochs short=50 full=120。
**不引入任何新超参数**（如更低微调学习率）——避免事后调参嫌疑；若 lr=1e-3 微调发散，按负结果
如实记录，运行中途禁止改参。

### 2.2 warm-start 实现语义

- `ALSimulationRunner` 新增 `init_from_ckpt: Optional[Dict]`（state_dict）；
  `_fit_stage` 在 `trainer.fit()` 前 `model.load_state_dict(init_from_ckpt)`——**每个预算点**
  都从 best.pt 起步（非链式继承上一阶段权重），与 W14「每点独立消除累积漂移」的设计对偶：
  本协议每点共享同一外部先验，差异只来自各自标注集。
- torch.manual_seed 仍在加载前置位：dataloader shuffle 顺序可复现；优化器 AdamW 全新构建
  （不加载 ckpt 的 optimizer state——那是全量训练收敛态的动量，对 20~200 样本微调是噪声源）。

### 2.3 打分器域内性规则（本实验核心修正）

- **熵打分器 = 上一累计预算的域内微调模型**（即上一 `_fit_stage` 返回的对象——它从 best.pt
  初始化、在偏移域已标注子集上微调过 ≥1 epoch）。
- **禁止原始 best.pt 直接跨域打分**：即使 best.pt 是各阶段微调的初始化来源，任何一次池上
  打分都必须发生在域内微调之后。首个增量（b: 20→50）的打分器 = b=20 微调产物
  ——这正是 W14 报告 next§2 「warm-start 打分器 + 域内少量标注校准」路线的实现。
- 断言方式（TDD）: monkeypatch 捕获每次 `predict_probs` 收到的模型权重，断言其 ≠ 原始
  init state_dict（证明打分发生在微调后）。

### 2.4 成功与失败判据（预写）

主指标: Δ(b) = mean_entropy − mean_random（best_val_acc，n_seeds=3）。
b=20 为两臂共享随机核，臂间恒等，不参与判定；主判定预算点 = {50, 100, 200}。

- **正证据候选（成功）**: 在 {50,100,200} 中 **≥2 个预算点**满足
  ① Δ(b) > 0 且 ② Δ(b) > max(std_e, std_r)（超出单臂种子噪声带）
  ③ 方向一致性: 该点 3 seeds 中 ≥2 个 seed 熵更高 → E5 正证据候选成立，报告如实给出
    效应量与不确定性；论文回填仍由后续窗口执行（W23 禁触 docs/paper）。
- **负结果（失败）**: 其余一切情形（含随机继续反超）→ 如实归档，E5 维持 PENDING，
  报告四步结构必须给出机理解释与下一步建议，禁止择优叙事。
- **过程熔断**: 任一轨迹出现 NaN 或全预算点 val_acc=0 → 终止该轨迹留证上报，禁止当场改参续跑。

## 3. 与 W14 结果的可比性声明

W23 曲线与 W14 曲线**绝对数值不可直接互比**：数据分布（0.10 vs 0.05）与数据实例
（seed 不同）都变了。可比的是**协议内部对比结构**：两窗内 entropy vs random 都是同数据、
同初始化来源、同超参的受控对照。跨报告只允许做定性方向性陈述（如「warm-start 是否扭转了
冷启动下随机反超的方向」），且必须注明分布不可比。三层口径铁律不受影响。

## 4. CLI 决策：新建 `scripts/run_p05_al_warmstart.py`（二选一之理由）

不采用在 `run_p05_al_efficiency.py` 上加 `--protocol` 开关，理由：

1. **W18 排队任务零风险隔离**: W18 看护脚本按字面命令启动 Q1
   （`run_p05_al_efficiency.py --config p05_al_full.yaml --fresh`）；共享入口改造会让排队任务
   与新协议共用一条代码路径，违反接力机制的最小惊讶原则。
2. **必填项差异**: warm 协议有额外必填（init ckpt 路径、诊断选定噪声档、scorer 规则），
   独立 config（`configs/p05_al_warmstart_{short,full}.yaml`）比开关+条件分支更可审计。
3. **可比性声明的物理保障**: 两入口 git 历史互不触碰，W14↔W23 协议对照可审计。

代价（如实登记）: 入口编排逻辑有 ~150 行相似代码。接受此重复以换取隔离性；
共享数学逻辑（采样器/打分/汇总）仍在 `psd/training/active_learning.py` 单一 owner 内复用。

---

### Task 1: TDD RED — warm-start 三断言

**Files:** Test: `psd/training/tests/test_active_learning.py` 追加

- [ ] 测试 `test_warm_init_loaded_at_every_stage`（SpyTrainer 捕获每阶段 fit 前权重 == init_from_ckpt）
- [ ] 测试 `test_offset_dataset_pipeline_deterministic_and_shifted`
  （20263/20264 + noise 0.10: 形状/确定性/与 0.05 数据实质不同/类别均衡）
- [ ] 测试 `test_scorer_is_in_domain_finetuned_not_raw_ckpt`
  （捕获 predict_probs 输入权重 ≠ 原始 ckpt 权重）
- [ ] 运行确认 FAIL

### Task 2: TDD GREEN — 实现

**Files:** Modify: `psd/training/active_learning.py`

- [ ] `ALSimulationRunner.__init__(..., init_from_ckpt=None)`；`_fit_stage` 加载逻辑
- [ ] pytest 全绿（含既有 17 条回归）→ commit

### Task 3: CLI + 双配置

**Files:** Create: `scripts/run_p05_al_warmstart.py`、`configs/p05_al_warmstart_short.yaml`、
`configs/p05_al_warmstart_full.yaml`

- [ ] 冒烟（tiny 覆盖端到端）→ commit

### Task 4: CPU short 全量扫描 + 归档

- [ ] 6 轨迹 × 4 预算点（50ep）跑完，JSON 含 per-seed 明细与 mean±std
      → `reports/p05-al-efficiency-warmstart-short-2026-08-25.json`
- [ ] 报告 `reports/w23-p05-al-warmstart-2026-08-25.md`（observe→interpret→implicate→next
      + 双向论证 + 负结果预案）→ commit

### Task 5: GPU 移交

- [ ] full config 备好（120ep/device auto/AMP；RUNDATE 占位）
- [ ] 按 W18 协议把启动命令追加进 GPU 接力队列（Q 编号顺延），本窗不自抢卡
- [ ] HANDOVER §8 回写 + 任务书状态更新 + stage-plan 登记 → commit

## Self-Review

- 任务书五步门全覆盖 ✓ / 判据预写无事后空间 ✓ / scorer 域内性可测 ✓ / CLI 二选一已说明理由 ✓
- 白名单核对: active_learning.py(+tests) / scripts/run_p05_al_* / configs/p05_al_* /
  plans w23 / reports w23-* 与 p05-al-efficiency-warmstart-* ✓
