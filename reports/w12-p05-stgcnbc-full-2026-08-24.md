# W12 窗口 — P0.5 骨干微调评估报告（合成层完整训练 + E6 双贴合实验）

> 日期: 2026-08-24 · 窗口: W12 · 任务书: `dev-docs/handovers/W12-p05-stgcnbc-full.md`
> 状态: ✅ 六项完成标准全达成
> **指标口径: 本报告全部数字属【合成层 synthetic】，与公开真实层 / 真实 K9 层严禁混报（AGENTS.md 规则#3）**

---

## 1. 执行摘要

在 W11 前置工程（440 样本冒烟基线 val_acc=18.2%）之上，W12 完成：

1. **扩量合成数据**：20→100 样本/类（440→2200 样本，19.0 MB），TDD 先测后码（6 绿）
2. **ST-GCN+BC 完整训练**：50 epoch + 早停 patience=15，n=100 主实验 **best_val_acc = 97.27% @ epoch 35**
3. **样本量消融**：20/50/100 三档 val_acc = 77.27% / 95.00% / 97.27%（单调递增，边际收益递减）
4. **E6 双贴合实验**：Y（22 类）97.27% vs Y′（21 类，stand+track→locomotion）95.91%，**粗粒度未受益（−1.36pp）**
5. **三层口径报告 + W13 移交**（本文 §6/§7）

---

## 2. Step 1 — 扩量合成数据生成

### 2.1 TDD 前置

`psd/data/tests/test_stgcn_bc_dataset_scale.py` 新增 6 测试：总量 2200、shape (30,24,3)、22 类完整性（每类恰 100）、seed=42 确定性、Dataset 包装、pickle 存取。**先红后绿**，最终 6 passed。

### 2.2 生成结果（实测）

```
路径: data/synthetic/syn_22class_100per_class_seed42.pkl
总样本: 2200 (22 类 × 100 样本/类)
文件大小: 19.01 MB
每个 clip: (T=30, V=24, C=3) np.float32
生成命令: .venv/Scripts/python.exe scripts/gen_synth_22class.py --samples-per-class 100 --output data/synthetic/syn_22class_100per_class_seed42.pkl
```

- `scripts/gen_synth_22class.py` 参数化改造（新增 `--samples-per-class/--output/--T/--seed` CLI），manifest 改为追加式列表登记
- `data/synthetic/_manifest.json` 追加 n=100 条目（gitignore 排除，不入库）
- `docs/DATA_LOCATIONS.md` §4 合成层小节更新为三行清单（n=50/n=20/n=100 实测数字）

---

## 3. Step 2 — ST-GCN+BC 完整训练 + 样本量消融

### 3.1 实验设置

配置 `configs/p05_stgcn_bc_full.yaml`（新建，不改 W11 文件）：lr=1e-3 / AdamW / cosine LR + warmup5 / batch 32 / AMP / epochs 50 / early stopping patience 15 / seed 42 / val_split 0.2。模型 1,433,759 参数（22 类）。入口脚本 `scripts/run_p05_full.py`（HANDOVER §8 白名单 `scripts/run_p05_*` 通配覆盖），支持 `--samples-per-class` 消融覆盖。

GPU 错峰：开工前 `nvidia-smi` 实测 2026 MiB / 8151 MiB，无 W9 Phase B NTU 残留；白天短训窗口内完成（单次 ≤7 min）。

### 3.2 消融结果（同一 config，仅变 samples_per_class）

| n/类 | 总样本 | train/val | best_val_acc | best_epoch | epochs_run | 早停 | 证据 JSON |
|------|--------|-----------|--------------|------------|------------|------|-----------|
| 20   | 440    | 352/88    | **77.27%**   | 31         | 46         | 触发 | `reports/p05-stgcnbc-synthetic-20perclass-Y.json` |
| 50   | 1100   | 880/220   | **95.00%**   | 11         | 26         | 触发 | `reports/p05-stgcnbc-synthetic-50perclass-Y.json` |
| 100  | 2200   | 1760/440  | **97.27%**   | 35         | 50         | 未触发（跑满）| `reports/p05-stgcnbc-synthetic-100perclass-Y.json` |

**样本量-精度曲线**：20→50 提升 +17.73pp，50→100 提升 +2.27pp——边际收益递减，50/类后接近合成任务饱和区。

### 3.3 主实验收敛曲线描述（n=100，runs/p05_stgcn_bc_full/history.json）

```
ep01 loss=2.0867 train_acc=36.4% val_acc=37.3%
ep05 loss=0.4517 train_acc=84.3% val_acc=88.0%
ep10 loss=0.2189 train_acc=93.8% val_acc=96.1%
ep15 loss=0.1494 train_acc=94.6% val_acc=96.6%
ep35 loss=0.0529 train_acc=99.8% val_acc=97.27% ← best
ep50 loss=0.0505 train_acc=100%  val_acc=96.6%
```

loss 单调下降趋势显著（首 1/3 均值 0.4209 > 尾 1/3 均值 0.0519），全程无 NaN。

### 3.4 收敛判据（config 写死三项，主实验全过）

| 判据 | 阈值 | 实测 | 结果 |
|------|------|------|------|
| loss 单调下降趋势 | first_third > last_third | 0.4209 > 0.0519 | ✅ |
| val acc 显著超随机基线 | ≥ 13.5%（4.5%×3） | 97.27% | ✅ |
| 无 NaN | all finite | finite | ✅ |

### 3.5 与 K9 参照对比（诚实披露）

| 数字 | 层 | 来源 |
|------|----|----|
| **97.27%** | 合成层（本仓，n=100/类） | 本次运行 |
| 46.97% @ ep21 | 合成层（K9 phase-3） | HANDOVER §7 速查表 |
| 18.2% @ ep5 | 合成层（W11 冒烟，5 epoch） | `reports/w11-p05-prep-2026-08-24.md` |

本仓合成层成绩**远超** K9 参照（97.27% vs 46.97%）。按任务书 §7 升级路径要求，如实披露根因假设、**不归因于方法缺陷**：

1. **任务难度不对称（主因假设）**：本仓合成集由 W11 重实现的"姿态模板 + 高斯噪声"生成器产出，train/val clip 共享同一姿态模板（仅噪声相位不同），本质是**模板匹配**任务；K9 phase-3 合成集的生成配置未知（可能含更强类间相似度或域偏移），直接对比不公平。
2. **训练配置未知**：K9 的 lr/batch/augmentation 不可考，无法控制变量复现。
3. **结论边界**：97.27% 仅用于合成层内部趋势分析（样本量曲线、双贴合对比），**不可外推**至公开真实层或真实 K9 层。原始数字保留供论文讨论段使用。

---

## 4. Step 3 — E6 双贴合分类体系实验

### 4.1 实验设计（冻结进 configs/p05_e6_taxonomy.yaml）

| 场景 | 标签体系 | 类数 | 映射 |
|------|---------|------|------|
| Y（细粒度·考核单） | 原始 22 类 | 22 | 不映射（assets-map §1 权威清单） |
| Y′（粗粒度·日报） | stand(2)+track(8)→locomotion | 21 | 其余 20 类不变，随机基线 1/21≈4.76% |

实现：`scripts/run_p05_full.py` 内置 `_Y_TO_YP_MAP` 冻结映射表 + `_map_samples_to_yprime()`；两体系同 seed=42、同切分比例、同超参，唯一变量为标签体系。

### 4.2 结果（n=100/类，当次运行证据）

| 体系 | 类数 | best_val_acc | best_epoch | epochs_run | 随机基线 | 超基线倍数 |
|------|------|--------------|------------|------------|----------|-----------|
| Y  | 22 | **97.27%** | 35 | 50（跑满） | 4.55% | 21.4× |
| Y′ | 21 | **95.91%** | 9  | 24（早停） | 4.76% | 20.2× |

**差值：Y′ − Y = −1.36pp —— 粗粒度合并未带来精度收益。**

### 4.3 分析（双向论证）

**为什么 Y′ 没有更准（反方质疑成立）：**
1. **类内方差增大**：locomotion = stand（基础站姿模板）+ track（前肢下压前伸模板）两个差异显著的模板合并，类内异质性上升，学习难度不降反升。
2. **合并方向未命中真混淆源**：本合成集中模板完全相同的是 stand↔stay（生成器中二者均为 pass 直通），Y 的上限误差来自这对孪生类；而 ADR 映射是 stand→locomotion，stay 仍单列——最易混对未被消解。
3. **训练动态混杂**：Y′ 于 ep9 达峰后 patience=15 在 ep24 早停；Y 持续训练至 ep35 达峰。部分差值可能来自有效训练时长而非体系本身。

**正方理由（粗粒度的理论收益为何未兑现）：**
理论上合并可分性低的类应提升准确率，但前提是**被合并类本身高频互混**；本合成集各类模板区分度高（Y 已达 97.27%，接近模板匹配饱和区），无混淆红利可释放——粗粒度收益在合成层无法显现，需真实层数据验证（归 W13+/论文窗口）。

### 4.4 口径差异披露（Y 的类数定义）

任务书 §3 表格中 Y 定义为 23 类（jump→jump_up/jump_down 对半拆分）；但用户开工指令、HANDOVER v1.7 §8 路由行、任务书 §6 完成标准三处一致定义为 **Y=22 类**。本窗口按后者执行。补充工程事实：当前合成生成器中 jump 仅单一姿态模板，若对半拆分为 jump_up/jump_down，两子类模板完全相同、原理上不可分——23 类变体须先扩展生成器模板，归论文窗口按需追加（任务书 §5 默认"不做三层粒度"一致）。

---

## 5. 边界合规确认

| 规则 | 状态 |
|------|------|
| 一切 `*smq*` 文件禁触（W4 领地活跃） | ✅ 未触碰 |
| K9 仓全程只读 | ✅ 零写操作 |
| mamba_ssm 不装 / Mamba 不移植（§7 升级路径保留） | ✅ 未安装未移植 |
| 三层口径严禁混报 | ✅ 全文合成层标注，§6 三层对照表 |
| GPU 错峰（白天短训 + 跑前 nvidia-smi） | ✅ 实测 2026/8151 MiB 无 W9 残留 |
| `psd/models/stgcn_bc_*.py` 只读不改 | ✅ 零改动（W11 落盘物保持原样） |
| Conventional Commits 中文 + 白名单暂存 | ✅ 见 §8 |

---

## 6. 三层指标口径对照表

| 层 | 数据 | 本阶段状态 | 关键数字 |
|----|------|-----------|---------|
| **① 合成层**（本窗口实验） | `syn_22class_*_seed42.pkl`（自产模板+噪声） | ✅ 完成 | 20/50/100 消融 77.27%/95.00%/97.27%；E6: Y 97.27% / Y′ 95.91% |
| **② 公开真实层** | InterPet4D smal_npy（225 clip 有效） | ⏳ 待 W13 InterPet4D 微调 | 参照锚点：P0.1 kNN 20.89%（2.51× 随机） |
| **③ 真实 K9 层** | 产品侧训练视频 | ⏳ 待产品侧提供 | **P0.5 ≥85%/22 类验收线适用于此层** |

> ⚠️ 合成层 97.27% ≠ 真实层可达精度：合成集 train/val 同模板的模板匹配特性决定其数值系统性偏高，仅作内部趋势证据。

---

## 7. W13 移交声明

| 交付项 | 路径/接口 | 说明 |
|--------|----------|------|
| 微调训练入口 | `psd.training.train_stgcn_bc.train(config_path) -> dict` | 返回 `{best_val_acc, best_epoch, total_epochs_trained, final_train_acc, final_val_acc, device, use_amp}` |
| 训练器底层接口 | `STGCNBCTrainer(model, train_samples: List[Dict], val_samples: List[Dict], config: TrainConfig)` | 接收原始样本 dict 列表（keypoints 为 numpy (T,24,3)），内部自行包装 DataLoader；真实层数据经 `load_pyskl_pickle()` 转换后即可喂入 |
| 双贴合映射参考 | `scripts/run_p05_full.py::_map_samples_to_yprime` + `configs/p05_e6_taxonomy.yaml` | Y′ 冻结映射表可直接复用于真实层评估 |
| 合成层基准数字 | `reports/p05-stgcnbc-synthetic-{20,50,100}perclass-Y.json` + `-Yprime.json` | 论文讨论段素材（样本量曲线 + 双贴合对比） |
| P0.5 验收口径澄清 | stage-plan "≥85%" | 该线属**真实 K9 层**验收线；合成层基准已闭环，W13 应聚焦公开真实层（InterPet4D）微调 |
| 遗留可选实验 | jump 拆分 23 类变体 / Y″ 中间粒度 | 归论文窗口按需（须先扩生成器模板） |

---

## 8. 提交清单（白名单核对）

```
configs/p05_stgcn_bc_full.yaml                    [新建]
configs/p05_e6_taxonomy.yaml                      [新建]
psd/data/tests/test_stgcn_bc_dataset_scale.py     [新建 TDD 6 绿]
scripts/gen_synth_22class.py                      [修改：CLI 参数化（Step 1 命令隐式授权）]
scripts/run_p05_full.py                           [新建（HANDOVER §8 scripts/run_p05_* 通配）]
docs/DATA_LOCATIONS.md                            [修改：仅 §4 合成层小节]
reports/w12-p05-stgcnbc-full-2026-08-24.md        [新建本文]
reports/p05-stgcnbc-synthetic-20perclass-Y.json   [新建]
reports/p05-stgcnbc-synthetic-50perclass-Y.json   [新建]
reports/p05-stgcnbc-synthetic-100perclass-Y.json  [新建]
reports/p05-stgcnbc-synthetic-100perclass-Yprime.json [新建]
dev-docs/stage-plan.md                            [修改：仅 P0.5 行]
data/* runs/*                                     [gitignore 排除不入库]
```

回归验证：全量测试 **172 passed**（W11 159 + 本窗号新增 + 其他窗口既有），零回归。

---

## 9. 完成标准核对（任务书 §6 六项）

- [x] 扩量合成集（100 样本/类）本地生成成功（2200 样本 / 19.01 MB），`DATA_LOCATIONS.md` 含实测数字
- [x] ST-GCN+BC 完整训练收敛：val_acc=97.27% 显著超随机基线（≥13.5%），loss 单调下降（0.42→0.05），无 NaN
- [x] E6 双贴合实验：Y（22 类）97.27% 和 Y′（21 类）95.91% 均有当次运行证据 JSON
- [x] 消融：20/50/100 样本/类三档 val_acc 对比（77.27% → 95.00% → 97.27%）
- [x] 报告归档 `reports/w12-p05-stgcnbc-full-2026-08-24.md`，含三层口径标注
- [x] 中文 Conventional Commit，只暂存白名单文件

---

*报告编制: W12 窗口（歆歆）2026-08-24*
*依据: AGENTS.md v1.0 / HANDOVER v1.7 §8 / W11 报告 §5 / ADR-0002 裁决③ v1.2 / stage-plan P0.5 行*
