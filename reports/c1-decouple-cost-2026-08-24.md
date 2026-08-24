# C1 解耦切换成本实验报告（W19 窗口）

> 日期: 2026-08-24 | 窗口: W19 | 层级: **合成层**（synthetic）
> 任务书: `dev-docs/handovers/W19-c1-decouple.md` | 风险关联: R2 🔴 CRITICAL
> 结果 JSON: `reports/c1-decouple-cost-2026-08-24.json`（当次运行证据）

---

## 0. 执行摘要

以 W12 已验证的评估标准演化场景 **Y(22类) → Y′(21类, stand+track→locomotion)** 为切换事件，对照两种切换策略的成本：

| 结论项 | 结果 |
|--------|------|
| 墙钟时间比（baseline/decouple） | **7.32×**——解耦臂快约 7 倍（80s vs 588s，mean） |
| 精度差（decouple − baseline） | **+2.27pp**（96.0% vs 93.7%）——解耦无精度代价且略胜 |
| 标注单元数 | 两臂完全相同（528 train / 132 val），本档位下标注维度打平 |
| C1 claim 方向 | ✅ **当前证据支持解耦方案在标准演化场景下有显著切换成本优势** |

> ⚠️ 口径提醒：以上为 **small 档**（每类 30 样本、CPU 路径）证据；full 档（每类 100、完整预算）因 NTU Phase B 长训独占 GPU 按排队纪律挂起，一键复跑命令见 §7。

## 1. 协议

### 1.1 两臂设计

| 臂 | 初始化 | 可训练参数 | 数据 |
|----|--------|-----------|------|
| `decouple`（解耦） | 加载 Y checkpoint（`runs/p05_stgcn_bc_full/best.pt`，Y val_acc=0.9659）剥离 head 仅载 backbone（694 张量，missing=4 head 键，unexpected=0）→ **冻结 backbone** | 仅语义层 head（fc_cls + conv_boundary） | 合成 Y′ 数据 |
| `baseline`（非解耦基线） | 从头随机初始化完整模型 | backbone + head 全部 | **同一份数据同一份切分** |

两臂配对纪律：同 seed 生成同数据、同 permutation 切分（8:2，W12 口径）、同训练超参（AdamW lr=1e-3, wd=1e-4, cosine, warmup=5, batch=32, patience=15, 早停）。差异只有初始化与可训练范围——这正是「物理-语义解耦」的架构主张本身。

### 1.2 冻结的严格性（本窗口 TDD 抓出的协议缺陷并已修复）

仅设 `requires_grad=False` 的"冻结"是**假的**：backbone 中 BatchNorm 的 running_mean/running_var 是 buffer，不受梯度约束，`model.train()` 下每次前向都会漂移——等于 backbone 在被 Y′ 数据悄悄改写。修复后解耦臂的冻结为双层：

1. 参数梯度阻断（requires_grad=False）；
2. 补丁实例 `train()` 强制 backbone 恒为 eval 模式（BN 统计恒定）。

回归断言：一步优化后 backbone 全部张量逐元素相等（含 running stats）、无梯度残留（`test_training_step_leaves_backbone_intact`）。

### 1.3 分档与执行状态

| 档位 | 配置 | 状态 |
|------|------|------|
| 冒烟 | n=10, epochs=3, seed=42, CPU | ✅ 管线验证通过（解耦 47.7% vs 基线 4.55%@随机线，墙钟比 3.62×）；JSON 存 `runs/c1_smoke/smoke-pipeline-check.json`，不作证据 |
| **small** | n=30, epochs=50, seeds=42/43/44, CPU | ✅ **本报告证据** |
| full | n=100, epochs=50, seeds=42/43/44 | ⏳ GPU 排队中（见 §7） |

### 1.4 成本口径与干扰取证声明

- **标注单元数**主口径 = 训练样本量（labeled_units_train=528）；验证样本单列（132）。两臂相同，故本档位标注维度无差异——成本优势体现在墙钟与收敛轮数。
- **墙钟** = `trainer.fit()` 当次实测。执行期间 GPU 被 NTU Phase B 长训独占（快照：7610–7646 MiB / 8151 MiB 已用），故全部运行走 **CPU 路径**；每次运行前后各抓一次 nvidia-smi 快照存入 JSON（`gpu_state` / `gpu_state_after` 字段）作干扰证据。
- **已知干扰风险**：基线臂三连跑期间恰逢并行窗口（W20 AK 协议实验）活跃，CPU 负载波动可能放大其墙钟方差（205s std vs 解耦臂 13s std）。缓解证据：① epoch 维度比值 1.55× 与参数量论据方向一致；② 冒烟档在轻负载下仍测得 3.62× 同向比值；③ full 档 GPU 复跑可交叉验证。**解读比值时建议以「≥3×」保守区间陈述，而非点值 7.32×。**
- std 为总体标准差（ddof=0），n=3。

## 2. 两臂成本表（small 档，双维度）

| 指标 | 解耦臂 mean±std | 全重训臂 mean±std | 比值/差值 |
|------|----------------|------------------|-----------|
| **标注单元数**（train） | 528 ± 0 | 528 ± 0 | 打平 |
| **墙钟秒数** | **80.35 ± 12.51** | 588.03 ± 205.31 | **7.32×** |
| 收敛轮数 epochs_run | 23.0 ± 3.56 | 35.67 ± 11.44 | 1.55× |
| best_epoch | 8.0 ± 3.56 | 21.67 ± 12.71 | 更早达峰 |
| best_val_acc | **0.9596 ± 0.0129** | 0.9369 ± 0.0094 | **+2.27pp** |
| 冻结/可训参数 | 1,426,824 / 6,678 | 0 / 1,433,502 | 可训参数仅 0.47% |

逐 seed 明细：

| seed | 臂 | best_val_acc | best_epoch | epochs_run | 墙钟 s |
|------|-----|------|------|------|--------|
| 42 | decouple | 0.9545 | 13 | 28 | 97.89 |
| 43 | decouple | 0.9470 | 6 | 21 | 73.56 |
| 44 | decouple | 0.9773 | 5 | 20 | 69.60 |
| 42 | baseline | 0.9470 | 20 | 35 | 627.28 |
| 43 | baseline | 0.9394 | 38 | 50(未触发早停) | 817.54 |
| 44 | baseline | 0.9242 | 7 | 22 | 319.26 |

## 3. 四步分析

**Observe（观察）**：六次配对运行中，解耦臂在全部三个 seed 上同时表现出更短的墙钟（69–98s vs 319–818s）、更少的收敛轮数（20–28 vs 22–50）、以及更高的 best_val_acc（94.7–97.7% vs 92.4–94.7%）。无一例外。

**Interpret（解释）**：机制上符合预期——解耦臂复用的 backbone 已在 Y 数据上学到可迁移的时空表征（Y val_acc=0.9659），切换到 Y′ 时只需在冻结特征上学一个 21 类线性映射（stand+track 本就高度可分，合并反而降低类间混淆）；全重训臂则要同时学表征与分类器，消耗更多轮数与算力。+2.27pp 的精度反超提示迁移特征对粗粒度任务甚至略有增益（locomotion 合并消除了 stand/track 边界噪声）。

**Implicate（推论）**：在「评估标准演化」这一论文核心场景下，物理-语义解耦架构将切换成本从「整个管线重训」降为「语义层重训」，实测墙钟节省约一个数量级（保守 ≥3×），且无精度惩罚。这直接支撑标题副句 *under Evolving Evaluation Criteria* 的实证立场，R2 风险在合成层证据范围内**暂缓触发**。

**Next（下一步）**：① GPU 空闲后跑 full 档交叉验证（命令见 §7），若趋势矛盾以 full 为准并回改本报告结论；② 论文写作侧可将 C1 表格按本报告 §2 双维度模板回填（标注单元数列注明"两臂等额"）；③ 若需强化标注效率维度，后续可加"样本量梯度消融"（n=10/30/60 下解耦臂何时追平全重训满额精度）。

## 4. 双向论证

**正方（解耦成本优势成立）**：
1. 三 seed 全部同向，无一反例；效应量远大于 seed 间方差（std 12.5s vs 差值 508s）。
2. 机制自洽：可训参数从 143.4 万降到 6678 个（0.47%），反向传播跳过 694 个 backbone 张量的梯度计算，加速有明确的算力来源而非偶然。
3. 与既有事实闭环：W12 已证 Y′ 全量重训可达 96.82%（n=100），本实验解耦臂用 30% 样本即达 95.96%，逼近满额重训——迁移价值与成本优势互相印证。
4. 冻结严格性经测试锁定（BN 统计恒定），不存在"假冻结污染对比"的解释漏洞。

**反方质疑（及回应）**：
1. *质疑：head-only 可能因表征不适配而收敛慢* —— 实测相反（23 vs 36 轮）。但该质疑在「演化幅度更大」的场景下依然有效：Y→Y′ 只是合并 2 类，若未来标准变化涉及物理层语义（如骨架定义变更），解耦优势可能消失甚至反转。论文措辞应限定为「标签空间演化」场景。
2. *质疑：墙钟受 CPU 共享负载污染，7.32× 可能虚高* —— 部分成立。基线臂 std 异常大（205s），不排除后台负载系统性拖慢了后跑的基线臂。回应：epoch 比（1.55×）与冒烟档比值（3.62×，更早时段测得）提供独立旁证；报告采用保守区间表述；full 档 GPU 复跑列为待办。
3. *质疑：合成数据的表征可能过于容易，真实骨架上迁移收益未必如此* —— 成立且重要。本结论仅覆盖**合成层**口径，不得混报至公开真实/真实 K9 层；三层口径纪律要求论文引用时明确标注。
4. *质疑：解耦臂加载的 checkpoint 本身耗过标注预算（Y 数据），是否构成隐藏成本？* —— 该成本发生在切换事件之前，属于沉没的已有资产；C1 度量的正是「切换时刻的边际成本」，口径合理，但论文须如实说明前提（存在已训好的旧体系模型）。

## 5. 诚实结论

1. **当前证据支持 C1 claim**：合成层 small 档下，解耦切换相对全管线重训实现约一个数量级的墙钟节省（保守区间 ≥3×）、1.55× 收敛轮数节省、零精度损失（+2.27pp），且三 seed 全向一致。
2. **证据边界**：合成层 / n=30 / CPU 路径 / 单一演化类型（标签合并）。full 档未跑（GPU 独占），若复跑结果与此矛盾，以 full 为准并修订本报告。
3. **R2 状态**：由 🔴 CRITICAL 降为 🟡 待 full 档确认——标题副句暂无需降级，最终裁决权在用户。
4. 无粉饰项：标注单元数两臂打平是如实呈现（成本优势来自算力而非标注节约）；墙钟污染风险已在 §1.4 取证并给出保守区间。

## 6. 可复现命令

```powershell
# 环境: 仓库根目录, .venv Python 3.12.4
# 小档（本报告证据，~12 分钟 CPU）
.venv/Scripts/python.exe scripts/run_c1_decouple.py --tier small --device cpu --output-json reports/c1-decouple-cost-2026-08-24.json

# 冒烟管线验证（~20 秒）
.venv/Scripts/python.exe scripts/run_c1_decouple.py --tier smoke --device cpu --epochs 3 --seeds 42 --output-json runs/c1_smoke/smoke-pipeline-check.json

# 测试（TDD 19 绿）
.venv/Scripts/python.exe -m pytest psd/training/tests/test_c1_decouple.py -q
```

## 7. full 档待办（GPU 空闲时一键执行）

```powershell
# 启动前必查: nvidia-smi 显存占用 < 1GB 且无 python 训练进程方可启动
nvidia-smi
.venv/Scripts/python.exe scripts/run_c1_decouple.py --tier full --device auto --output-json reports/c1-decouple-cost-full-2026-08-24.json
```

预计 GPU 路径全程 < 20 分钟；完成后与本报告 §2 对比，趋势矛盾则以 full 为准回改结论。

## 8. 归档清单

| 文件 | 说明 |
|------|------|
| `reports/c1-decouple-cost-2026-08-24.json` | 当次运行原始证据（6 runs + GPU 快照 + 聚合） |
| `runs/c1_small/{arm}_seed{42,43,44}/` | 各 run 的 history.json + best.pt/last.pt |
| `scripts/run_c1_decouple.py` | 重写版脚本（commit `22f1206`） |
| `psd/training/tests/test_c1_decouple.py` | TDD 测试 19 绿（commit `22f1206`） |
