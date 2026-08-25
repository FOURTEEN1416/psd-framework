# tab3「−自监督预训练」消融 full 档判读报告

> 执行窗口: **W39**（wt/W39，2026-08-26 00:21–00:30） | 设计与脚本来源: W31（只读执行，未改动本体）
> 模式: **full（GPU, RTX 5060 Laptop 8GB）** | 指标口径: **合成层**（best_val_acc, 22 类, W12 口径 2200 样本）
> warm 臂初始权重来源: **公开真实层**（P0.1 AimCLR @ InterPet4D, epoch120_model.pt）——两层口径分别披露，禁止混报（AGENTS.md 硬规则 3）
> 自动产物: `reports/w31-ablation-pretrain-2026-08-26.json` / `.md`（脚本硬编码命名）+ 本判读报告（任务书指定命名）

## 1. 设计与执行摘要

| 要素 | 值 |
|------|-----|
| 数据 | 合成 100×22=2200 样本（W12 口径, T=30, data_seed=42, 切分 8:2 → train 1760 / val 440） |
| 两臂 | scratch（随机初始化）vs warm（strict 加载 P0.1 AimCLR encoder_q）；同 seed 下头初始化逐位相等、切分与洗牌一致，**唯一差异 = encoder 初始权重**（TDD 断言保证） |
| 预算 | 50 epochs × batch 32, lr 1e-3, 早停关闭（等预算公平对照） |
| 规模 | 2 臂 × 3 seeds = 6 runs, 全部 device=cuda 完成, stderr 零异常 |
| 占卡依据 | relay state.json=ALL_DONE + W33 线性评估 00:13 正常收官（joint best 74.30%, Δ−0.04pp vs 官方）+ GPU 实测空闲；见 BOARD [08-26 00:25] |

## 2. 结果（best_val_acc）

| 臂 | mean | std | per_seed (s0/s1/s2) | best_epoch (s0/s1/s2) |
|----|------|-----|---------------------|------------------------|
| scratch | 96.82% | ±0.37pp | 96.82 / 96.36 / 97.27 | 30 / 7 / 28 |
| warm | 96.97% | ±0.28pp | 97.05 / 96.59 / 97.27 | 19 / 15 / 11 |

**方向判定 Δ = mean(warm) − mean(scratch) = +0.15pp**（n=3, 两臂 std 均 >Δ, 不显著）。
同 seed 配对差（唯一差异变量受控）: **+0.23 / +0.23 / 0.00 pp —— warm 2 胜 1 平 0 负, 方向完全一致但幅度在噪声带内**。

## 3. 四步分析

**Observe（观察）**：六次运行全部收敛至 96.4–97.3% 窄带；warm 臂三 seeds 的 best_epoch（19/15/11, mean 15.0±4.0）系统性早于 scratch（30/7/28, mean 21.7±12.9），且配对方向 2 胜 1 平零负。最终精度差 +0.15pp 小于 seeds 间波动。

**Interpret（解释）**：合成层 2200 样本 / 22 类对 ST-GCN+BC 已是饱和区（两臂 >96%, 训练 acc 同达 ~96%）——任务难度不足以区分表征初始化质量，天花板效应吞没了预训练的精度贡献。warm 臂收敛更早更稳是符合迁移学习文献的典型次级信号（encoder 已含可分特征, 分类头只需微调), 但在本预算下两臂最终都摸到同一天花板。

**Implicate（推论）**：① 本消融在合成层 full 档给出的诚实结论是「下游精度维度: 预训练收益 ≈ 0 (+0.15pp n.s.)」——它**不构成 C3 主张的反驳**, 因为 C3 的对照物是表征空间质量（kNN 20.89% = 2.51×随机, reports/p01-knn-result.json）而非微调终点精度, 两者是预训练价值的两个不同观测面; ② 论文若引用本行, 措辞必须限定「合成层饱和档位」, 并把预训练价值主张锚定在 kNN 表征证据 + 收敛动力学, 不得写成「预训练提升下游 X pp」; ③ 真正可能暴露预训练差距的是低资源档位（样本越少, 表征先验越值钱）——当前档位测不出≠不存在。

**Next（下一步）**：① tab3 该行按 §5 素材回填, 状态 PENDING→✅（合成层档位）, 注明「低资源/真实域档留后续批次」; ② 若审稿或叙事需要强消融数字, 最小增量实验 = samples_per_class ∈ {5,10,20} 梯度 × 两臂 × 3 seeds（纯 GPU 数分钟级, 复用本管线零代码改动, 仅改 config）——是否执行留协调者/用户裁决; ③ W36 终稿窗口引用本报告时同步引用 p01 kNN 证据链, 防止单一证据面误导。

## 4. 双向论证

**正方（预训练价值成立, 结果可安心入册）**：
1. 配对方向完美一致（2胜1平0负）且 warm std 更小（0.28 vs 0.37pp）——方向信号真实存在, 只是幅度被天花板压缩;
2. 收敛动力学优势显著（best_epoch 提前 ~6.7 ep 且方差缩小 3 倍）——等预算协议下这是真实的效率收益;
3. kNN 表征证据独立成立（20.89%=2.51×随机）, 预训练管线的科学价值不受本档位零增益影响;
4. 工程依赖事实: warm-start 是 C1 解耦架构与 W23 叙事换轨的组件, 消融证明它「至少无害」（+0.15pp 方向为正）即足以支撑架构选择。

**反方（该消融近乎空手, 引用需克制）**：
1. +0.15pp 在 n=3 下无统计意义（符号检验 p=0.25 双侧）, 写进正文任何「优于」句式都会被审稿人击穿;
2. 合成层非论文主战场——主战场是真实域低资源场景, 本档位结论外推性存疑, 反方可以说「你只在最不可能出差异的地方做了消融」;
3. 若论文 tab3 需要的是「去掉预训练性能崩塌」式戏剧性证据, 本行给不出——C3 的原始假设（kNN 掉回随机）从未被直接检验（那需要对 scratch encoder 做 kNN 对照, 本实验未设计该臂）;
4. 天花板效应意味着本实验的信息量主要在「饱和区无差异」这一否定性结论上, 单独成行价值有限, 必须搭配梯度档位才有完整故事。

## 5. tab3 回填素材（建议措辞, 归 experiment-skeleton owner 终审）

> **− 自监督预训练** | ✅ 可填（合成层 full 档, 2026-08-26）: warm-init vs scratch 下游精度 **Δ=+0.15pp（96.97%±0.28 vs 96.82%±0.37, n=3 seeds, 同 seed 配对 2胜1平, n.s.）**——合成层饱和档位（两臂均 >96%）下预训练对微调终点精度无可分辨贡献; 收敛动力学方向一致占优（best_epoch 15.0±4.0 vs 21.7±12.9）。预训练价值主张锚定 **kNN 表征质量（P0.1: 20.89%=2.51×随机）** 与解耦架构组件依赖, 非下游精度增幅。【合成层 / 指标; warm 权重来源公开真实层 P0.1】来源: `reports/ablation-pretrain-2026-08-25.md` §2–§4 + `reports/w31-ablation-pretrain-2026-08-26.json`
> ⚠️ 引用纪律: 本行不得改写为「预训练提升/降低下游精度」; C3 原假设（kNN 掉回随机）仍属未直接检验, 如需闭环须补 scratch-encoder kNN 对照臂。

## 6. 复现命令与证据清单

```bash
# full 档（GPU; 需 runs/p01_aimclr_pretext/epoch120_model.pt 就位）
& "D:\Desktop\psd-framework\.venv\Scripts\python.exe" scripts/run_ablation_pretrain.py --config configs/ablation_pretrain.yaml
```

| 证据 | 路径 |
|------|------|
| 逐 run 明细 JSON（6 runs 全字段） | `reports/w31-ablation-pretrain-2026-08-26.json` |
| 训练全程 history（6×50ep, 含 loss/lr/duration） | `runs/w31_ablation_pretrain/{scratch,warm}_seed{0,1,2}/history.json`（gitignore 资产随收编 -f 入库, 沿 W31 冒烟先例） |
| 点火日志 stdout/stderr | `runs/w39_ablation_stdout.log` / `_stderr.log`（stderr 空 = 零异常） |
| 占卡依据 | BOARD [08-26 00:25] + `D:/Desktop/psd-framework-W33/reports/ntu-phaseB-lineareval-2026-08-25.json` |
| 设计与 TDD | W31 交付 57d5203: psd/models/aimclr_finetune.py + psd/training/ablation_pretrain.py（13 绿断言含两臂初始化逐位相等/同 seed 同切分） |

---
*执行: W39 窗口（歆歆）· 2026-08-26 · 报告版本 1.0*
