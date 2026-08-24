# W11 交接文档 — P0.5 前置工程：assets-map 补链 + 合成数据移植重建 + ST-GCN+BC 进仓

> **你是 W11 窗口**。读完本文档即开工。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md` v1.5 → **`dev-docs/decisions/0002-user-rulings-ntu-synthetic-e6.md` 裁决② v1.1（你的存在依据）** → `docs/DATA_LOCATIONS.md` → K9 仓 `backend/ml/behavior/` 目录树（只读）。

---

## 1. 任务目标（一句话）

按 AGENTS.md 规则#6 显式移植流程，把 K9 仓合成数据生成器与 ST-GCN+BC 骨干训练栈迁入本仓，本地重建**带 22 类标签的合成骨架集**并登记入档——为 P0.3 Phase B 类别映射（消费者一）与 P0.5 微调 ≥85%/22 类（消费者二）备齐全部输入。**本窗口不做微调达标实验（W12 主实验窗口的事）。**

## 2. 直接输入（全部有据）

1. **ADR-0002 裁决② v1.1（用户拍板，锁定不可回退）**：路径 a = 移植 K9 合成数据生成器本地重建；触发点 = P0.5 开工前（即本窗口）；产物登记 `DATA_LOCATIONS.md` 后供 P0.3 Phase B 与 P0.5 消费。
2. **K9 仓资产指针（HANDOVER §6）**：
   - ST-GCN+BC 训练栈：`D:\Desktop\k9-training-system\backend\ml\behavior\stgcn_bc\`（合成 best_val_acc 46.97% @ epoch21，K9 phase-3）
   - Mamba 基线：`D:\Desktop\k9-training-system\backend\ml\behavior\`（合成 85.61%；⚠️ 具体文件名未确认——你编制 assets-map 时落实，**但 Mamba 本窗口不移植不安装**）
3. **⚠️ 阻断发现（2026-08-24 复核会话实锤）**：`docs/assets-map.md` 全仓不存在，但被 AGENTS.md 规则#6、HANDOVER §6、ADR-0002 裁决② 三处引用为移植流程 owner——truth 链断链。**你的 Step 0 就是补上它，否则后续一切移植无据可依。**
4. **环境事实**：torch 2.11.0+cu128 就绪（RTX 5060 Laptop 8GB，sm_120 实测）；`psd/models/` 当前为空（仅 .gitkeep）；mamba_ssm 未安装（刻意，见 §7）。
5. **P0.4 移交池（W10 已落盘，本窗口只登记不动用）**：`data/processed/p04/pseudo_pool_main_consensus_a1.0_seed42.jsonl`（191 条 @ iter3）——真实数据侧微调信号归 W12 消费。

## 3. 执行链

### Step 0：编制 `docs/assets-map.md`（阻断解除，一切之前）

- 只读遍历 K9 仓 `backend/ml/behavior/`（含 `stgcn_bc/` 子目录），列出合成数据生成器与 ST-GCN+BC 栈的完整文件清单。
- 登记映射表，每行五列：`源文件(K9 相对路径) → 目标路径(本仓) → 职责 → 移植方式(复制适配/重实现) → 测试锚点(对应 psd tests 文件)`。
- 从生成器源码提取 **22 类清单原文**，作为附录落进 map（这是全仓第一份 22 类权威记录，DATA_LOCATIONS 与 p05 config 均引用它，禁止另抄一份产生重复 truth）。
- 头部声明 owner 地位：本文档是跨仓代码复用的唯一映射 truth（对齐 AGENTS.md 规则#6）。

### Step 1：移植合成数据生成器 → 本地重建

- 目标落位：适配/封装代码进 `psd/data/synth_*.py`（外部实现逻辑按 assets-map 方式移植，不改 K9 原文件——K9 仓全程只读）。
- TDD：先写测试（生成的骨架张量形状/帧长分布/22 类标签域断言/随机种子确定性）到 `psd/data/tests/test_synth_*.py`，再移植实现。
- 生成本地重建集 → `data/synthetic/`（大文件 gitignore，目录内放 `_manifest.json`：样本数/类别分布/生成命令/种子）。
- 登记 `docs/DATA_LOCATIONS.md` 新增「合成层」小节：路径 + 数量级实测数字（防 W2 式"规划期估计"失真，必须实测后填写）+ 生成复现命令。

### Step 2：移植 ST-GCN+BC 训练栈

- 落位：模型定义 `psd/models/stgcn_bc.py`；训练管线 `psd/training/train_stgcn_bc.py`；配置 `configs/p05_stgcn_bc.yaml`（数据路径指向 Step 1 产物，22 类数从 assets-map 附录引用）。
- TDD：先测后码——前向输出形状 (B, 22)、损失有限且可下降一步、dataloader 标签域 ⊆ 22 类清单、CPU 小批量端到端一轮不炸。
- 接口约定（供 W12 直接消费）：`train_stgcn_bc.train(config_path) -> dict`，返回含 `best_val_acc / epochs_run / ckpt_path` 的结果字典；checkpoint 存 `runs/p05_stgcn_bc/models/`。

### Step 3：合成层冒烟训练（GPU 短任务，非达标线）

```bash
# 冒烟：小 epoch 验证收敛方向与量级（对照 K9 参照 46.97%，不设达标线——达标判定属 W12）
.venv/Scripts/python.exe scripts/run_p05_prep_smoke.py --config configs/p05_stgcn_bc.yaml --smoke
```

- 判据（写死进 config）：loss 单调下降趋势可见 + val acc 显著高于 22 类随机基线 4.5%（如 ≥10% 即方向正确）+ 无 NaN。
- GPU 纪律：单卡 RTX 5060 8GB；与 W9 Phase B 错峰默认=白天短冒烟/夜间长训练互斥让行；跑前 `nvidia-smi` 确认无残留进程。

### Step 4：归档移交

- `reports/w11-p05-prep-<日期>.md`：assets-map 编制结论 + 数据集实测数字 + 冒烟证据 + 一条命令复现序列。
- 向 W12（P0.5 主实验）移交：22 类合成集路径 + 可训 ST-GCN+BC 栈 + 冒烟基线数字；移交声明落报告。

## 4. 边界

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `docs/assets-map.md`（新建）、`psd/data/*synth*`(+tests)、`psd/models/**`、`psd/training/*stgcn*`(+tests)、`scripts/*p05*`、`configs/p05*`、`reports/w11-*`、`docs/DATA_LOCATIONS.md` 仅「合成层」小节、stage-plan 仅「P0.5 行状态列」 |
| ❌ 禁触 | 一切 `*smq*`（W4 领地，E-C 定稿归其 owner）、`docs/paper/**`、`dev-docs/decisions/**`（只读）、`external/**` 内部实现、`dev-docs/handovers/W1-W10` 既有文档、`.venv` 结构变更、K9 仓任何写操作 |
| GPU | 仅 Step 3 冒烟短任务；mamba_ssm 安装明确不做 |

## 5. 用户输入需求

| 时点 | 事项 | 默认 |
|------|------|------|
| 开工前 | GPU 排序确认 | 白天 W11 冒烟短任务 / 夜间 W9 Phase B NTU 训练，错峰互斥让行即可，无需专门裁决 |
| ~~W12 开工前~~ ✅ **已定稿** | E6 双贴合场景——用户 2026-08-24 "按照最优解进行"，正式采纳预注册候选：粗报 standing+walking+running→locomotion / 细考 jump 拆 jump_up·jump_down | 无待办；W12 直接消费 `ADR-0002 v1.2 裁决③终稿`，勿再询问用户 |

## 6. 完成标准

- [ ] `docs/assets-map.md` 落盘，且 Step 1/2 的每条移植都能在 map 中找到对应行（非摆设文档）
- [ ] 22 类合成骨架集本地生成成功，`DATA_LOCATIONS.md` 含实测数量级数字与复现命令
- [ ] ST-GCN+BC 本仓测试全绿（含既有测试零回归）+ Step 3 冒烟有当次运行证据
- [ ] 报告归档 `reports/` + 中文 Conventional Commit
- [ ] W12 移交声明落报告

## 7. 升级路径

- K9 生成器依赖缺失或无法脱离原环境运行 → 记录阻塞项，评估最小重实现工作量后**上报用户裁决**（对应 ADR-0002 裁决②回退预案条款：届时论文口径可能需改 7 类体系，须用户确认，不得自行降级）。
- Mamba 一切相关诉求（安装 mamba_ssm / 移植基线）一律记录不顺手做——是否双骨干对比属 W12+ 用户决策，Windows+sm_120 编译风险已知（WSL 经验在 K9 仓）。
- 发现 22 类清单在 K9 侧存在多版本 → 以 assets-map 附录登记版为准并在报告中披露差异，不得自行合并。

---

*交接编制: 歆歆（验收复核+规划会话）2026-08-24 · 依据: HANDOVER v1.5 §8 / ADR-0002 v1.1 裁决② / stage-plan P0.5 行 / 2026-08-24 复核会话阻断发现（assets-map 缺失）*
