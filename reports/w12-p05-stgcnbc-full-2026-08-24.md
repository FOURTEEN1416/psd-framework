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

> **方法论注记（复核补充披露）**：①三档数据集由同一生成器、同 seed=42 但不同随机消耗序列独立产生，**非嵌套子集关系**（仅首类样本重合）——"样本量↑精度↑"趋势结论不受影响，但论文引用曲线时应注明采样方式；②初版三次消融共用 `runs/p05_stgcn_bc_full` 导致 history/best.pt 相互覆盖（现盘仅存 n=100 曲线），关键数字以各档 JSON 归档为准；复核后脚本已修复：`--samples-per-class` 覆盖时自动追加 `_n{n}` 独立输出目录。

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

**初版（早停配置 patience=15）**：

| 体系 | 类数 | best_val_acc | best_epoch | epochs_run | 随机基线 | 超基线倍数 |
|------|------|--------------|------------|------------|----------|-----------|
| Y  | 22 | **97.27%** | 35 | 50（跑满） | 4.55% | 21.4× |
| Y′ | 21 | 95.91% | 9  | 24（早停） | 4.76% | 20.2× |

**复核优化（等预算重跑，消除训练时长混杂）**：Y′ 以 `--no-early-stopping` 跑满 50 epoch，三次独立运行（torch 种子未固定，见 §4.3 方差说明）：

| 运行 | best_val_acc | best_epoch | epochs_run | 证据 |
|------|--------------|------------|------------|------|
| #1 | 96.36% | 26 | 50 | runs history.json 交叉验证 |
| #2 | 95.91% | 23 | 50 | 会话 stdout（产物后被并行进程覆盖） |
| #3 | **96.82%** | 20 | 50 | commit `8f0c293` 归档 JSON（含 ckpt_path） |

**差值结论：等预算下 Y′ − Y ≈ −0.45 ~ −1.36pp（三运行均值 −0.91pp）——粗粒度合并未带来精度收益，方向在全部 5 次配对观测中一致。**

### 4.3 分析（双向论证）

**为什么 Y′ 没有更准（反方质疑成立）：**
1. **类内方差增大**：locomotion = stand（基础站姿模板）+ track（前肢下压前伸模板）两个差异显著的模板合并，类内异质性上升，学习难度不降反升。
2. **合并方向未命中真混淆源**：本合成集中模板完全相同的是 stand↔stay（生成器中二者均为 pass 直通），Y 的上限误差来自这对孪生类；而 ADR 映射是 stand→locomotion，stay 仍单列——最易混对未被消解。
3. **训练动态混杂（已在等预算重跑中消解）**：早停版 Y′ @ep9 达峰即停贡献了约一半表观差距；等预算下差距收窄但仍存在。

**正方理由（粗粒度的理论收益为何未兑现）：**
理论上合并可分性低的类应提升准确率，但前提是**被合并类本身高频互混**；本合成集各类模板区分度高（Y 已达 97.27%，接近模板匹配饱和区），无混淆红利可释放——粗粒度收益在合成层无法显现，需真实层数据验证（归 W13+/论文窗口）。

**运行方差披露（诚实边界）**：本管线仅固定 numpy 种子（数据生成+切分），DataLoader shuffle 与模型初始化走 torch 全局未播种 RNG，单次运行方差实测 ±0.5pp 量级（等预算三次运行 95.91/95.91/96.82）。当前"Y > Y′"方向性结论跨全部观测成立，但**差距幅度（≈1pp）与方差同量级**，定量化需多种子重复实验（≥5 seeds 报均值±标准差）——归论文窗口，本窗口不做（GPU 预算约束 + 结论方向已稳）。

**第三方并行运行记录**：复核期间检测到并行进程使用本脚本产出过一次 95.68%@ep15（30ep 早停变体）运行并覆盖过证据文件一次（详见 §5 并发事件）——该数值同样落在上述方差带内，进一步佐证结论稳健性。

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

### 并发事件记录（复核发现，透明披露）

| 时点 | 事件 | 处置 |
|------|------|------|
| 11:59-12:01 | 并行进程两条提交先砍后补 `psd/models/stgcn_bc_constants.py`（`003704b`/`decf490`） | 本窗实验最晚落盘 11:57:44 未受波及；当前 HEAD 全量复测 172 passed |
| ~12:30 | 本窗等预算 Y′ 重跑产物 JSON（96.36%@26）被并行进程回滚至 HEAD 旧版（内容/mtime 证据确凿）；runs/ 下 history.json 因 gitignore 幸存并交叉验证了数字 | 以幸存 history.json + stdout 双证恢复结论；最终以受控重跑 #3（秒级链式提交）重生成权威证据 |
| 12:47+ | 第三方进程用本脚本跑出 95.68%@ep15（早停变体）再次覆盖证据文件 | 数值落在方差带内不影响结论；以链式提交夺回产物控制权 |
| 提交 `8f0c293` | 共享 git index 竞争：并行进程预先暂存的 W4 领地文件 `p02-smq-iou-eC-seeds-recheck.json` 被连带扫入本窗提交 | 不重写共享历史（并行者活跃，rebase 风险大）；如实披露，该产物归属 W4 owner 不变 |

> 教训沉淀：多窗口并行时证据产物应在生成后**秒级提交**；跨窗口共享的 runs/reports 路径建议在任务书中预先约定互斥或带窗口后缀。

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

提交记录：
- `0a2df69` feat(p05): W12 合成层骨干微调评估完成（12 文件）
- `8f0c293` fix(p05): 复核优化——ckpt_path 落档 + 等预算开关 + 消融独立输出目录 + Y′ 等预算证据重生成
  - ⚠️ 该提交连带扫入并行进程预暂存的 `reports/p02-smq-iou-eC-seeds-recheck.json`（W4 领地，见 §5 并发事件表），产物归属不变

回归验证：全量测试 **172 passed**（W11 159 + 本窗号新增 + 其他窗口既有），零回归；外来 constants 提交后当前 HEAD 复测通过。

---

## 9. 完成标准核对（任务书 §6 六项）

- [x] 扩量合成集（100 样本/类）本地生成成功（2200 样本 / 19.01 MB），`DATA_LOCATIONS.md` 含实测数字
- [x] ST-GCN+BC 完整训练收敛：val_acc=97.27% 显著超随机基线（≥13.5%），loss 单调下降（0.42→0.05），无 NaN
- [x] E6 双贴合实验：Y（22 类）97.27% 和 Y′（21 类）均有当次运行证据 JSON（早停版 95.91% @ ep9 + 等预算三运行 95.91~96.82%，见 §4.2）
- [x] 消融：20/50/100 样本/类三档 val_acc 对比（77.27% → 95.00% → 97.27%）
- [x] 报告归档 `reports/w12-p05-stgcnbc-full-2026-08-24.md`，含三层口径标注
- [x] 中文 Conventional Commit，只暂存白名单文件

---

## 10. 复核优化记录（2026-08-24 第二轮）

子智能体派发两度网络失败，降级为结构化自查（逐项新鲜证据验证，不信记忆）。发现并处置：

| # | 问题 | 严重度 | 处置 |
|---|------|--------|------|
| 1 | E6 对比训练预算不等（Y′ 早停@24 vs Y 跑满 50） | Important | `--no-early-stopping` 等预算开关 + 三次重跑，差距 −1.36pp → −0.45~−1.36pp（方向不变） |
| 2 | 结果 JSON 缺 ckpt_path 字段（证据链断点） | Minor | 脚本补写 ckpt_path + early_stopping_effective 进 summary；已随 commit `8f0c293` 归档验证 |
| 3 | 消融三档共用 output_dir 相互覆盖 | Minor | 脚本修复为独立 `_n{n}` 目录；历史覆盖已在 §3.2 披露 |
| 4 | 死代码（未用 `_y_idx`、重复赋值行） | Minor | 已清除 |
| 5 | torch 种子未播种 → ±0.5pp 运行方差 > E6 表观差距 | Important | §4.3 方差披露；多种子重复归论文窗口 |
| 6 | 消融非嵌套采样未说明 | Minor | §3.2 注记补充 |
| 7 | 并行进程回滚证据文件 + index 竞争扫入 W4 文件 | Important（流程） | §5 并发事件表如实披露；秒级提交策略沉淀 |
| 8 | 外来提交后未复测全量测试 | Minor | 当前 HEAD 复测 **172 passed** ✓ |

**E6 结论修订**：等预算下粗粒度仍无收益（全部配对观测方向一致），但幅度与运行方差同量级——报告口径从"−1.36pp"改为"−0.45~−1.36pp，方向稳健、幅度待多种子定量化"。

---

*报告编制: W12 窗口（歆歆）2026-08-24 · 复核优化: 同日第二轮*
*依据: AGENTS.md v1.0 / HANDOVER v1.7 §8 / W11 报告 §5 / ADR-0002 裁决③ v1.2 / stage-plan P0.5 行*
