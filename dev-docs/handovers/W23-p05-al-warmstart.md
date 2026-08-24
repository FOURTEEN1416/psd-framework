# W23 任务书 — P0.5-AL Warm-Start 协议实验（用户裁决 A2 · 2026-08-24）

> 状态: ⏳ 待启动
> 前序状态: W14 冷启动协议负结果已归档（`reports/w14-p05-al-efficiency-2026-08-24.md`：b≥100 随机反超熵 7~8pp；best.pt 真实池打分 softmax 饱和退化）
> 裁决依据: 用户 2026-08-24 于歆歆协调会话拍板 **A2（warm-start + 加噪偏移）**，授权以本册为准
> 口径: 全程**合成层**（AGENTS.md 三层铁律；公开真实层路线归 W20，禁止混报）

## 0. 一句话目标

验证「熵不确定性采样 vs 随机采样」在 warm-start 设定下的效率差异，产出 22 类 val_acc 效率曲线（预算 {20,50,100,200} 片段 × 3 seeds，mean±std 误差棒）——E5/fig4 的正证据候选；**负结果同样如实归档，禁止择优叙事**。

## 1. 执行序（①是②的门，禁止跳步）

### Step 1 诊断选噪声档（CPU ~10min，先于一切实现）
- 生成偏移数据：池 seed=20263 / val seed=20264，noise_std ∈ {0.10, 0.15, 0.20} 各一套（spc 与 W14 相同：池 10/类、val 15/类）
- best.pt（`runs/p05_stgcn_bc_full/best.pt`）零微调直接评估：记录各档 val_acc 与 logit top1−top2 边际
- **选档判据**：基线 val_acc 落于 40%~70% 且未饱和（边际均值 < 10）
- **熔断判据（预写）**：所有档位基线 >85%（无适应空间）或全部饱和 → 实验不可行，停手留证上报用户重选方案，不得自行换招

### Step 2 预注册（落盘后才许写正式扫描代码）
- 文档：`docs/superpowers/plans/2026-08-XX-w23-warmstart-al.md`
- 必含：选定噪声档及诊断数据表、协议（每预算点从 best.pt 初始化微调；**熵打分器 = 上一累计预算的域内微调模型，禁止原始 best.pt 直接跨域打分**）、预算/seeds{42,43,44}/epochs(short 50/full 120)、成功与失败判据、与 W14 冷启动结果的可比性声明（仅策略维度变化）

### Step 3 TDD 实现
- 复用 `psd/training/active_learning.py` 的 ALSimulationRunner 骨架，新增 `init_from_ckpt` 参数（warm 初始化）
- 新增测试先行：warm 初始化生效断言、偏移数据管线断言、scorer 域内性断言
- CLI 复用 `scripts/run_p05_al_efficiency.py` 加 `--protocol warmstart` 或新建 `run_p05_al_warmstart.py`（二选一，说明理由）

### Step 4 短预算全量扫描 + 归档
- CPU 跑 short（50ep × 6 轨迹），曲线 JSON 含误差棒归档 `reports/p05-al-efficiency-warmstart-short-*.json`
- 报告 `reports/w23-p05-al-warmstart-2026-08-XX.md`（四步结构 observe→interpret→implicate→next + 双向论证）

### Step 5 GPU 移交（禁止自抢显卡）
- full-budget(120ep) 配置备好后，**按 W18 任务书协议把启动命令追加进 GPU 接力队列**（Q 编号顺延），不在本窗直接跑长任务

## 2. 领地与禁触

- 白名单：`psd/training/active_learning.py(+tests)`、`scripts/run_p05_al_*`、`configs/p05_al_*`、`docs/superpowers/plans/*w23*`、`reports/w23-*`、`reports/p05-al-efficiency-warmstart-*`、HANDOVER/stage-plan 回写行
- 禁触：`docs/paper/**`（当前由 W21/W22 占用——你若做出正证据，论文回填由后续窗口做）、`dev-docs/decisions/**`、`*ntu*`、`external/**`
- 并行纪律：产物落盘即提交；写 tracked 文件前必 `git diff` 重读（W14 曾发生 4 次并行覆盖事故）

## 3. 与相邻窗口的边界

| 窗口 | 关系 |
|------|------|
| W14（已完成） | 提供冷启动基线与全套可复用代码；本实验是协议维度的对照升级 |
| W18 GPU 队列 | 冷启动 full-budget 已排 Q1（对照组）；你的 full 排队顺延，科学判读不出队 |
| W20 公开真实层 | 三层口径的另一层；互不引用对方数字 |

## 4. 验收门

- [ ] 诊断表 + 预注册文档先于实现提交
- [ ] pytest 全绿（新增测试先行）
- [ ] 曲线 JSON 含 per-seed 明细与 mean±std
- [ ] 报告含双向论证与负结果预案
- [ ] HANDOVER §8 本行状态回写 + 修订历史登记

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-24 | 建册（用户裁决 A2 生效版） |
