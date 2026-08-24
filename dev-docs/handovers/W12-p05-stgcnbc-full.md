# W12 交接文档 — P0.5 骨干微调评估（合成层完整训练 + E6 双贴合实验）

> **你是 W12 窗口**。读完本文档即开工。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md` v1.6 → **`reports/w11-p05-prep-2026-08-24.md`**（你的直接输入）→ `configs/p05_stgcn_bc.yaml`（W11 落盘基线配置）→ `docs/assets-map.md`（22 类权威附录）。

---

## 1. 任务目标（一句话）

在 W11 前置工程之上，**完成 ST-GCN+BC 合成层完整训练 + E6 双贴合分类体系实验 + 三层指标口径报告**，为 P0.5 达标判定提供合成层基准，同时关闭 ADR-0002 裁决③ v1.2 E6 实验设计遗留项。

---

## 2. 直接输入（全部有据，无需二次确认）

1. **W11 移交物**（`reports/w11-p05-prep-2026-08-24.md` §5.1）：
   - 合成骨架集：`data/synthetic/syn_22class_20per_class_seed42.pkl`（440 样本，20/类）
   - ST-GCN+BC 训练栈：`psd/models/stgcn_bc_*.py` + `psd/training/train_stgcn_bc.py`
   - 冒烟基线：val_acc=18.2% @ epoch 5，loss 单调下降，无 NaN
   - 接口约定：`train(config_path)->dict` 已就绪

2. **K9 合成参照**（HANDOVER v1.6 §7 速查表）：
   - ST-GCN+BC 合成 best_val_acc = **46.97%** @ epoch 21（K9 phase-3）
   - 22 类随机猜测基线 = **4.5%**

3. **ADR-0002 裁决③ v1.2（已定稿，无需再问用户）**：
   - **粗粒度 Y′（日报场景）**：stand + track → **locomotion**；其余 20 类不变
   - **细粒度 Y（考核单场景）**：完整 22 类；jump → **jump_up / jump_down**（拆分为 2 类，共 23 类）
   - 类别映射的工程细节冻结进实验配置，不在本 ADR 展开

4. **stage-plan P0.5 验收线**：≥85%/22 类（⚠️ 该线适用于**真实 K9 层**，非合成层；合成层报告用 K9 参照 46.97% 对齐，禁止混报——AGENTS.md 规则#3）

5. **GPU 错峰**：白天短任务优先；跑前 `nvidia-smi` 确认无 W9 Phase B NTU 残留。

---

## 3. 执行链四步

### Step 1：扩量合成数据生成（关键瓶颈突破）

**背景**：W11 冒烟仅用 20 样本/类（共 440 样本），模型 143 万参数严重欠拟合。K9 参照 46.97% 在更大合成集上测得。W12 第一要务是扩量。

```bash
# 生成 100 样本/类（2200 总量），保持 seed=42 可复现
.env/Scripts/python.exe scripts/gen_synth_22class.py --samples-per-class 100 --output data/synthetic/syn_22class_100per_class_seed42.pkl
```

- 目标文件：`data/synthetic/syn_22class_100per_class_seed42.pkl`（gitignore 排除）
- 同步更新 `data/synthetic/_manifest.json`（追加 n=100 行）
- 同步更新 `docs/DATA_LOCATIONS.md` 合成层小节（实测数量）
- TDD 前置：写 `psd/data/tests/test_stgcn_bc_dataset_scale.py` 验证 n=100 时 shape/类别完整性

### Step 2：ST-GCN+BC 完整训练 + 收敛分析

更新 `configs/p05_stgcn_bc.yaml` → `configs/p05_stgcn_bc_full.yaml`（新建，不改 W11 文件）：

```yaml
# P0.5 ST-GCN+BC 完整训练配置（W12 窗口）
# 口径: 合成层（synthetic）
# 参照: K9 phase-3 合成 46.97% @ epoch 21

data:
  synthetic_path: data/synthetic/syn_22class_100per_class_seed42.pkl
  T: 30
  seed: 42
  val_split: 0.2

model:
  in_channels: 3
  num_classes: 22
  base_channels: 64
  num_stages: 10

train:
  lr: 0.001
  weight_decay: 0.0001
  epochs: 50                  # 比 W11 冒烟 30 更多
  batch_size: 32
  use_amp: true
  device: auto
  early_stopping: true
  patience: 15                # K9 使用 20，本仓稍激进
  output_dir: runs/p05_stgcn_bc_full

# 达标判定（非硬线——合成层 ≠ 真实 K9 层）
evaluation:
  synthetic_reference: 0.4697 # K9 phase-3 合成 46.97%
  random_baseline: 0.045      # 22 类随机猜测
```

**扩展 run 脚本**（复用 W11 `run_p05_prep_smoke.py` 架构，新增 `--full` 模式）：

```bash
# 完整训练（GPU，~50 epoch，预计数分钟）
.venv/Scripts/python.exe scripts/run_p05_prep_smoke.py --config configs/p05_stgcn_bc_full.yaml
```

**收敛判据（写死 config）**：
- loss 单调下降趋势可见
- val acc > 随机基线 4.5% × 3 = 13.5%（显著超基线）
- 无 NaN
- 早停触发时打印 best_epoch + best_val_acc

**消融实验（同一 config，不同 samples_per_class）**：
```bash
# ablation: 20样本/类（W11 基线复现）
.venv/Scripts/python.exe scripts/run_p05_prep_smoke.py --config configs/p05_stgcn_bc_full.yaml --samples-per-class 20

# ablation: 50样本/类
.venv/Scripts/python.exe scripts/run_p05_prep_smoke.py --config configs/p05_stgcn_bc_full.yaml --samples-per-class 50

# ablation: 100样本/类（主实验）
.venv/Scripts/python.exe scripts/run_p05_prep_smoke.py --config configs/p05_stgcn_bc_full.yaml --samples-per-class 100
```

结果以 JSON 归档至 `reports/p05-stgcnbc-synthetic-{n}perclass.json`。

### Step 3：E6 双贴合分类体系实验

**实验设计**（冻结进 config，非临场拍脑袋）：

| 场景 | 标签体系 | 类数 | 映射说明 |
|------|---------|------|---------|
| Y（细粒度·考核单） | 原始 22 类，jump 拆分为 jump_up/jump_down | 23 | jump 类样本按姿态模板一半赋 jump_up、一半赋 jump_down |
| Y′（粗粒度·日报） | standing+track→locomotion，其余 20 类不变 | 21 | 合并 stand(2)+track(8) 为 locomotion |

**实现要点**：
- Y′ 映射表冻结进 `configs/p05_e6_taxonomy.yaml`
- 重跑 Step 2 训练（Y′ 版本），对比 Y vs Y′ 的 val_acc 差异
- 报告：两类体系的 acc 差值 + 分析（粗粒度是否显著受益）

**关键 constraint**：E6 实验仅在合成层运行（符合 AGENTS.md 三层口径），不得与公开真实/真实 K9 层混报。

### Step 4：报告归档 + 移交 W13

- `reports/w12-p05-stgcnbc-full-2026-08-24.md`：
  - 合成层训练结果（val_acc / K9 参照对比 / 收敛曲线图描述）
  - E6 双贴合实验结果（Y vs Y′ acc 差值）
  - 三层口径标注：**合成层**（本窗口实验）、**公开真实层**（待 W13 InterPet4D 微调）、**真实 K9 层**（待产品侧提供）
  - W13 移交声明（InterPet4D 微调入口 + P0.5 验收口径澄清）

---

## 4. 边界

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `configs/p05_stgcn_bc_full.yaml`、`configs/p05_e6_taxonomy.yaml`、`psd/data/stgcn_bc_dataset.py` 仅追加扩量函数、`psd/data/tests/test_stgcn_bc_dataset_scale.py`、`scripts/run_p05_prep_smoke.py` 仅追加 `--samples-per-class` 参数、`reports/w12-*`、`reports/p05-stgcnbc-synthetic-*.json`、`docs/DATA_LOCATIONS.md` 仅合成层小节更新、`stage-plan` 仅 P0.5 行 |
| ❌ 禁触 | 一切 `*smq*`（W4 领地）、`docs/paper/**`（论文窗口）、`dev-docs/decisions/**`（只读）、`external/**` 内部实现、`psd/models/stgcn_bc_*.py`（W11 已落盘，仅可读不改）、K9 仓任何写操作、mamba_ssm 安装 |
| GPU | 仅 Step 2 完整训练；与 W9 Phase B NTU 错峰（白天优先，跑前 nvidia-smi 确认） |

---

## 5. 用户输入需求

| 时点 | 事项 | 默认 |
|------|------|------|
| 开工前 | GPU 排序确认 | 白天 W12 短训（预计 <5min）/ 夜间 W9 Phase B NTU 训练，错峰互斥让行 |
| Step 3 E6 实验后 | 是否追加 Y'' 中间粒度（3 层粒度消融） | 默认不做——W12 只跑 Y 和 Y′ 两档；Y'' 归论文窗口按需追加 |

---

## 6. 完成标准

- [ ] 扩量合成集（100 样本/类）本地生成成功，`DATA_LOCATIONS.md` 含实测数字
- [ ] ST-GCN+BC 完整训练收敛：val acc 显著超随机基线 4.5%（≥13.5%），loss 单调下降，无 NaN
- [ ] E6 双贴合实验：Y（22 类）和 Y′（21 类，stand+track→locomotion）均有当次运行证据
- [ ] 消融：20/50/100 样本/类三档 val_acc 对比曲线
- [ ] 报告归档 `reports/w12-p05-stgcnbc-full-2026-08-24.md`，含三层口径标注
- [ ] 中文 Conventional Commit，只暂存白名单文件

---

## 7. 升级路径

- 扩量生成器依赖缺失 → 记录阻塞项，评估最小重实现工作量后**上报用户裁决**
- 合成层 val acc 远低于 K9 参照 46.97% → 如实报告差距 + 根因分析（样本不足？数据增强不足？），**不归因于方法缺陷**（K9 训练配置未知，直接对比不公平）
- 发现 W11 冒烟结果与 K9 参照差异过大 → 在报告中披露，**不自行调参追平**，保留原始数字供论文讨论段使用

---

*交接编制: 歆歆（规划会话）2026-08-24 · 依据: HANDOVER v1.6 §8 / W11 报告 §5 / ADR-0002 裁决③ v1.2 / stage-plan P0.5 行*
