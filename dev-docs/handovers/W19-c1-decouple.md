# W19 任务书 — C1 解耦切换成本实验

> 窗口: W19（全新独立窗口，不依赖任何旧窗口会话）
> 日期: 2026-08-24 | 编制: 协调者歆歆
> 前序状态: W15 窗口留下 `scripts/run_c1_decouple.py` 草稿（**未测试、未执行、零证据**），本窗口接管重做
> 关联: `dev-docs/rescue-plan-2026-08-24.md` §1 提示词 A 的文档化版本

## 1. 任务目标（一句话）

量化「物理-语义解耦」在评估标准演化场景下的成本优势：以 W12 已验证的 Y(22类)→Y′(21类 locomotion 合并) 为切换场景，对比【解耦方案：冻结 backbone 仅重训语义层】vs【非解耦基线：全管线重训】的标注单元数与墙钟时间，为论文 C1 claim 与标题副句 *under Evolving Evaluation Criteria* 提供唯一证据。

> ⚠️ 本实验直接对应风险登记册 R2（🔴 CRITICAL）：若结论不利，论文标题需降级为 "...for Low-Resource Animal Behavior Recognition"（用户裁决项）。

## 2. 背景事实（无需再调研）

- Y→Y′ 映射已在 W12 验证：`reports/p05-stgcnbc-synthetic-100perclass-Yprime.json` best_val_acc=0.9682，两次运行完全可复现
- 精度维度已知：Y′ 比 Y 低 1.36pp（粗粒度未受益）——本实验只补**成本维度**
- 映射代码参考：`scripts/run_p05_full.py` 中 `_Y_TO_YP_MAP`（stand/track→locomotion 合并）
- 草稿脚本：`scripts/run_c1_decouple.py`（已收编入库但**未经 TDD**；其映射逻辑正确，可作为起点但必须测试后才能信任）

## 3. 执行步骤

### Step 1 — TDD 先行
- 测试文件: `psd/training/tests/test_c1_decouple.py`
- 必测点:
  - `_Y_TO_YP_MAP` 正确性（22 类全覆盖，stand/track→同一 locomotion idx，其余一一对应）
  - backbone 冻结断言（重训后 backbone 参数梯度为零/参数不变）
  - 成本记录字段完整性（标注单元数、墙钟秒数、epoch 数、best_val_acc）
- 全绿后才允许写主逻辑

### Step 2 — 实验设计（两臂对照）
| 臂 | 配置 | 记录 |
|----|------|------|
| 解耦臂 | 加载 Y checkpoint → 冻结 backbone → 仅 head 在 Y′ 数据上训练 | 标注单元数(=head 训练样本量)、墙钟、epochs、best_val_acc |
| 非解耦基线 | 同数据从头训练完整模型（backbone+head） | 同上 |
- 数据: 合成 22 类集按 W12 口径生成/加载，映射到 Y′ 后使用（每类样本量用 `--n-per-class` 参数化）
- seeds ≥3（seed=42,43,44），报 mean±std
- 成本口径声明：若墙钟实测受 GPU 共享干扰，须记录当时 GPU 占用状态并在报告注明

### Step 3 — 分档执行（GPU 排队纪律）
```powershell
# 冒烟档（CPU 或抢卡间隙，~分钟级）：--n-per-class 10 --epochs 3
# 小档：--n-per-class 30
# full 档（GPU 空闲时）：--n-per-class 100 完整 epochs
```
- 启动前必查: `nvidia-smi` 显存占用。NTU Phase B 长训（PID 见当次查询）独占期间只跑冒烟/小档或纯 CPU 路径
- full 档结果与冒烟档趋势矛盾时，以 full 为准并报告差异

### Step 4 — 报告与归档
- `reports/c1-decouple-cost-<日期>.md`: 含执行摘要、协议、两臂成本表、四步分析（observe→interpret→implicate→next）、双向论证（正方：解耦应省成本 / 反方质疑：head-only 可能因表征不适配而收敛慢）、诚实结论
- 结果 JSON: `reports/c1-decouple-cost-<日期>.json`
- 若结论不利（解耦无成本优势或精度崩坏）：如实登记 + 在报告中给出「标题降级预案」供用户裁决，**禁止粉饰**

### Step 5 — 回写
- `dev-docs/stage-plan.md` P0.6 行或风险 R2 相关行做行级替换更新（编辑前重读最新内容）
- commit 信息: `feat(p06): C1 解耦成本实验——<一句话结论>`（Conventional Commits 中文）

## 4. 领地边界（互斥纪律）

**可写**: `scripts/run_c1_decouple.py`、`psd/training/tests/test_c1_decouple.py`、`configs/c1_*`、`reports/c1-*`、`runs/c1_*`
**禁触**: `docs/paper/**`、`dev-docs/decisions/**`、`*ntu*` 文件、`dev-docs/HANDOVER.md`（协调者统一回写）、`data/**`（只读）、`external/**`

## 5. 完成标准

- [ ] TDD 全绿且全仓 pytest 回归不劣于当前基线（≥189 passed）
- [ ] 两臂 × ≥3 seeds 有当次运行 JSON 证据
- [ ] 报告含成本双维度 + 双向论证 + 可复现命令
- [ ] 结论无论正负均如实归档

## 修订历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-24 | 独立窗口版任务书建册（接管 W15 未完成草稿） |
