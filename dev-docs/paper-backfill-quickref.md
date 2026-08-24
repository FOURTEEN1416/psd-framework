# 论文回填速查表

> Owner: `dev-docs/paper-backfill-quickref.md`（单一真相）
> 日期: 2026-08-24
> 用途: P0.5 完成后回填 `docs/paper/experiment-skeleton.md`，禁止从各报告零散翻数字

---

## 实验数字总表（三层口径分离，禁止混报）

### 合成层（Synthetic Layer）

| 实验 ID | 口径 | 指标 | 数值 | 随机基线 | 倍率 | 来源 | 论文 skeleton 行 |
|---------|------|------|------|---------|------|------|-----------------|
| E1 | ST-GCN+BC / 22类 / 100样本/类 | val_acc | **97.3%** | 4.5% | 21.6× | `reports/p05-stgcnbc-synthetic-100perclass-Y.json` | 复核 **96.4%**（08-24 重跑 · 早停 ep31/best ep16 · 21.4×） |
| E2 | ST-GCN+BC / 21类(E6-Y') / 100样本/类 | val_acc | **95.9%** | 4.76% | 20.1× | `reports/p05-stgcnbc-synthetic-100perclass-Yprime.json` | 复核 **95.7%**（08-24 重跑 · 早停 ep30/best ep15） |
| E3 | ST-GCN+BC / 22类 / 50样本/类 | val_acc | 95.0% | 4.5% | 21.1× | `reports/p05-stgcnbc-synthetic-50perclass-Y.json` | (消融) |
| E4 | ST-GCN+BC / 22类 / 20样本/类 | val_acc | 77.3% | 4.5% | 17.2× | `reports/p05-stgcnbc-synthetic-20perclass-Y.json` | (消融) |
| E-C | SMQ / 端到端 K=8 / 种子伪GT | mean_matched_iou | **0.458 ± 0.049** | ~0.30 | 1.53× | `reports/p02-smq-iou-eC-seeds.json` | 复核 **0.458 ± 0.049**（08-24 重跑 · F1=0.343 · `reports/p02-smq-iou-eC-seeds-recheck.json`） |
| E-A | SMQ / mse=1.0 修复基线 / 种子伪GT | mean_matched_iou | 0.409 | ~0.30 | 1.36× | `reports/p02-smq-iou-eA-seeds.json` | (历史) |

### 公开真实层（Public-Real Layer）

| 实验 ID | 口径 | 指标 | 数值 | 随机基线 | 倍率 | 来源 | 论文 skeleton 行 |
|---------|------|------|------|---------|------|------|-----------------|
| P0.1 | AimCLR / kNN(k=1) / InterPet4D smal_npy / dog-ID 代理 | top-1 | **20.89% ± 4.45%** | 8.33% | 2.51× | `reports/p01-knn-result.json` | [PENDING] |
| P0.3 | 姚青 JIA Phase A / 原型聚类 | purity | **0.534** | 0.330(Σπ²) | 1.62× | `reports/p03-jia-phasea-2026-08-24.md` §4.2 | [PENDING] |
| P0.3 | 姚青 JIA / 30% 标签噪声消融 | purity@30% | **0.503 ± 0.006** | — | 1.52× | 同上 | (消融) |
| P0.4 | TCL 迭代自训练 / 物理先验共识真值 / cov=35% | pool_precision | **0.691 ± 0.013** | 0.513(r0) | 1.35× | `reports/p04-tcl-2026-08-24.md` | [PENDING] |
| P0.4 | TCL / 保守首末配对检验 | Δpool_precision | **+10.69 ± 3.28 pp** (p=0.030) | — | — | 同上 | (统计) |

> ⚠️ 口径披露要求：每个数字必须标注"合成层/公开真实层"，禁止跨层混排表格。
>
> 🔁 **复核约定（2026-08-24）**：P0.2 / P0.5 已按下方「回填命令序列」当次重跑；「论文 skeleton 行」列标注『复核』的为重跑新鲜值，「数值」列保留首跑数字作对照。E1/E2 的来源报告文件已被重跑覆写（seed=42 固定，±0.9pp 内差异为 GPU 非确定性波动）；E-C 首跑报告保留不动，新鲜值落 `-recheck.json`。结论均不变：E1 96.4%、E2 95.7% 远超随机基线×3 判据线；E-C 与首跑逐位一致。
>
> 🛡️ 运维注记：首份 E-C recheck 文件（12:30 写入）曾被并行执行体删除（疑似 12:37 计划任务 `\OpenCode\psd-overnight-supervisor` 实例偏离任务书），已确定性重跑再生并立即入库。建议晨会核查该计划任务与并行窗口的写权限边界。

---

## 实验配置快照（复现用）

### P0.1 AimCLR
```
checkpoint: runs/p01_aimclr_pretext/epoch120_model.pt
kNN: k=1, backbone 256d, 5-fold CV, seed=42
数据: InterPet4D smal_npy (225 clips, T=64 re-sampled, NTU 视图)
```

### P0.2 SMQ（E-C 定稿）
```
checkpoint: runs/p02_smq_eC/models/epoch-30.model
配置: configs/p02_smq_eC.yaml（mse_loss_weight=1.0, patch_size=16, num_actions=8）
评估: scripts/eval_smq_segmentation.py --gt-protocol seeds
指标: mean_matched_iou + boundary_F1@16 + 随机基线对照
```

### P0.3 姚青 JIA（Phase A）
```
配置: kmeans×ratio×K 扫描，最优 K 对应 purity 0.534
噪声消融: q∈{0,10,20,30}%，30% 仅降 3.1pp
```

### P0.4 TCL
```
配置: calib✓ consensus✓ α=1.0，预算≤6 轮
指标: 池精度（物理先验共识真值），cov=35% 操作点
```

### P0.5 ST-GCN+BC（合成层）
```
checkpoint: runs/p05_stgcn_bc_full/best.pt（Y 22类）/ runs/p05_stgcn_bc_Yprime/best.pt（Y' 21类）
配置: configs/p05_stgcn_bc_full.yaml / configs/p05_e6_taxonomy.yaml
数据: data/synthetic/syn_22class_100per_class_seed42.pkl（2200 样本，8:2 训练/验证）
E6 双贴合: stand(2)+track(8)→locomotion，21 类
```

---

## 回填命令序列

```powershell
# 1. 运行所有实验的复现命令（验证数字新鲜度）
.\.venv\Scripts\python.exe scripts/eval_smq_segmentation.py --config configs/p02_smq_eC.yaml --iou --ckpt runs/p02_smq_eC/models/epoch-30.model --gt-protocol seeds --out reports/p02-smq-iou-eC-seeds-recheck.json
.\.venv\Scripts\python.exe scripts/run_p05_full.py --config configs/p05_stgcn_bc_full.yaml
.\.venv\Scripts\python.exe scripts/run_p05_full.py --config configs/p05_e6_taxonomy.yaml

# 2. 提取数字并回填 skeleton
# 手动从 reports/*.json 读 aggregate.mean_matched_iou / summary.best_val_acc
# 更新 docs/paper/experiment-skeleton.md 所有 [PENDING] 占位符
```

---

*编制: 歆歆 2026-08-24 · 依据: 各 P0 报告 + ADR-0004 走法 A · 复核回填: 歆歆 2026-08-24（P0.2/P0.5 当次重跑，证据见 reports/）*
