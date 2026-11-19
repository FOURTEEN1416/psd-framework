# P1 Sprint 证据归档清单（reports/p1_sprint_evidence/）

> 归档时间：2026-11-16 · 操作：证据归档员（只复制/新建，未删除任何文件，未执行 git add/commit）
> 来源：`runs/p1_sprint/`（该目录被 `.gitignore:19 /runs/*` 忽略，故将可提交证据镜像至本目录）
> 完整性校验：全部文件已用 `cmp` 与源文件逐字节比对，**全部一致**（E1–E3 三份 summary 于 2026-11-16 02:04 补充归档；E3b k 阶梯 summary 于同日统稿时补充归档；**2026-11-16 终审 P1 收口前刷新**：`k_ladder_summary.json` 发现为补测前旧快照（03:10:01，k≤50 单种子 42），已用 `runs/p1_sprint/k_ladder/summary.json`（17:52:40 canonical 全档位三种子合并版）重新覆盖，`cmp` 逐字节一致；同批新增补测会话镜像 `k_ladder_summary_s43_44_k102550.json`（源 `runs/p1_sprint/k_ladder/summary_s43_44_k102550.json`，17:51:03 落盘），`cmp` 逐字节一致）。
> **2026-11-16 终审回修补充归档（真实 tier 冻结编码器对照）**：`frozen_real_tier/` 子目录新增 3 份 summary（ntu120/ucf101/panaf500，源 `runs/p1_sprint/{ntu120,ucf101,panaf500}/summary.json`，13:04–14:01 执行），逐份 `cmp` 逐字节一致。

## 一、归档文件（23 个 + 本清单 MANIFEST.md）

### 实验结果 summary.json（9 个，逐字节校验通过）

| 归档路径 | 源路径 | 关键内容 |
|---|---|---|
| `merge_rename_upper/summary.json` | `runs/p1_sprint/merge_rename_upper/summary.json` | merge/rename 零重训上界（ntu60→ntu120 10 类，seed 42，预算 10%=4009 clips）：rename-oracle 上界 top1 **0.7676**，argmax-vote 下界 0.7651，目标全监督参照 0.715，retention 1.0735 |
| `standard_selftrain_baseline/summary.json` | `runs/p1_sprint/standard_selftrain_baseline/summary.json` | 标准自训练基线（τ=0.95，10 iters）：std-ST top1 0.6639 vs 参照 linear10 0.6605，Δ=+0.0034；含 τ∈{0.8,0.9} 敏感性；耗时 5899.9s |
| `smq_downstream_ablation/summary.json` | `runs/p1_sprint/smq_downstream_ablation/summary.json` | SMQ 下游消融（225 clips，7 伪标签类）：with_smq 0.4595 vs without_smq 0.4595，**Δ=0.0**；耗时 48.1s |
| `frozen_backbone_contrast/summary.json` | `runs/p1_sprint/frozen_backbone_contrast/summary.json` | 冻结三臂对照（50 clips/class，seed 42，100 epochs）：**frozen 0.6864 / 176.9s，fullft 0.9818 / 342.3s，scratch 0.9773 / 365.7s**（frozen_verified=true，可训练参数 6,423） |
| `zerobound_6049_summary.json` | `runs/p1_sprint/zerobound_6049/summary.json` | E1：TRANS-002 同 episode（60→49 ten-merge）零重训上界重测（seed 42，10% 标注头预算 4,009 clips，纯推理 112.5s，零重训零新标注）：rename-oracle 上界 **0.6804**，argmax-vote 下界 0.6797，源 60 类头 0.6605，目标 49 类全监督参照 0.6756（retention 1.007） |
| `semantic_rebuild_1100_summary.json` | `runs/p1_sprint/semantic_rebuild_1100/summary.json` | E2：TRANS-002 三臂 @ 1,100-clip 对照预算（seeds 42/43/44，80 epochs，预算抽取 seed 42）：C′ 0.4493 / 180.0s，D′ **0.6016 / 6.8s**，M 0.4820 / 10.2s；D′−C′ = +15.23pp，成本比 **26.47×** |
| `split_addition_episode_summary.json` | `runs/p1_sprint/split_addition_episode/summary.json` | E3：合成层 22→29 split/addition 迁移 episode（3 拆分 + 4 新增，seeds 42/43/44，50 clips/class，60 epochs）：remap 0.4299 → decoupled(k=10) 0.4575 → frozen head 0.6805 → coupled **0.9713**；新类精度阶梯 + 中位成本比 **3.02×**（verdict PARTIAL） |
| `k_ladder_summary.json` | `runs/p1_sprint/k_ladder/summary.json` | E3b：同 episode 定义标注预算阶梯 k∈{10,25,50,100}（episode seed 42；**全档位三种子 42/43/44**，2026-11-16 17:52:40 canonical 合并版，含 17:36 补测会话合并；池扩到 100/类）：decoupled **0.4529**±0.0016 / 0.5908±0.0252 / 0.6506±0.0133 / **0.7241**±0.0000，coupled 0.9586±0.0049（k 无关参照），frozen head 0.7172±0.0049；coupled−decoupled 边际 **50.57/36.78/30.80/23.45pp** 全部在 ±2.3pp 平价带外 → **k\*=None（verdict=no_parity_within_tested_range）**；k=10 档 100 vs 1000 新标注、92.1s vs 234.0s（seed 42；三种子 92.1/92.5/93.1 vs 234.0/232.6/229.1） |
| `k_ladder_summary_s43_44_k102550.json` | `runs/p1_sprint/k_ladder/summary_s43_44_k102550.json` | E3b 补测会话原件（seeds 43/44 × k∈{10,25,50} × decoupled/frozen\_head 两臂；coupled 臂 k 无关按种子复用；17:51:03 落盘）：canonical 合并版 `k_ladder_summary.json` 的输入之一，随终审一并归档留证 |

### 真实 tier 冻结编码器对照 summary（3 个，2026-11-16 终审回修归档，`frozen_real_tier/` 子目录，逐字节校验通过）

| 归档路径 | 源路径 | 关键内容 |
|---|---|---|
| `frozen_real_tier/ntu120_summary.json` | `runs/p1_sprint/ntu120/summary.json` | NTU120 frozen-encoder 三臂（12,000-clip/60 类子集，10 seeds）：C 0.5434 / A 0.4542 / B **0.4836 均值**（b_arms），retention **0.8886 PARTIAL**；trainable_params=0（总参 930,620），weights_provenance=retrained_same_protocol_saved |
| `frozen_real_tier/ucf101_summary.json` | `runs/p1_sprint/ucf101/summary.json` | UCF101 frozen-encoder 三臂（10,000-clip 子集，官方 split1 val 463 clips，10 seeds）：C 0.2311 / A 0.1404 / B 均值 0.1540，retention **0.6664 FAILS**；trainable_params=0（总参 941,157） |
| `frozen_real_tier/panaf500_summary.json` | `runs/p1_sprint/panaf500/summary.json` | PanAf500 frozen-encoder 三臂（500 clips/9 类，30 seeds）：C 0.60 / A 0.52 / B 均值 0.5147，retention **0.8578 PARTIAL**；trainable_params=0（总参 1,240,073） |

三份 summary 的 full_ref/linear_10pct/b_arms/retention/verdict 与 2026-09-07 已报告 E9 产物（reports/p5b-ntu120|p5b-ucf101|p23-panaf-retention-2026-09-07.json）逐位一致；weights_provenance 与 caveat 如实随件。协议全文：supplementary.tex app:frozenreal；正文载体：04-experiments.tex ssec:boundary + tab:attribution-controls 真实 tier 块。运行日志（ntu120/ucf101/panaf500 的 .log/.err.log）留在 `runs/p1_sprint/` 未镜像。

### 运行日志 *.log（11 个，顶层平铺；E1–E3 的运行日志仍在 `runs/p1_sprint/` 未镜像，见第四节第 4 条）

`autopilot.log`、`autopilot_run.log`、`frozen_backbone_contrast.attempt1.log`、`frozen_backbone_contrast.log`、`frozen_backbone_contrast.py.log`、`merge_rename_upper.log`、`merge_rename_upper.py.log`、`smq_downstream_ablation.log`、`smq_downstream_ablation.py.log`、`standard_selftrain_baseline.log`、`standard_selftrain_baseline.py.log`

## 二、novelty survey log 位置核验

**在仓内，均为 git 已跟踪文件（`git ls-files --error-unmatch` 通过，且未被 .gitignore 忽略）：**

1. `dev-docs/research/NOVELTY_CHECK_YAOQING_JIA.md` —— 姚青 JIA 方法创新性核验报告（W1 窗口，2026-08-23；GitHub 工具链检索矩阵 10 组查询，结论：首次性初步成立）
2. `dev-docs/handovers/W1-novelty-check.md` —— W1 创新性核验任务交接文档
3. 辅助定位材料：`dev-docs/research/PAPER_POSITIONING.md`（论文定位，含 novelty 相关内容）

## 三、建议的 git add 白名单（供用户确认后提交，本归档员未执行）

```bash
# A. 本证据目录（新增，18 文件 + 本清单）
git add reports/p1_sprint_evidence/

# B. P1 sprint 实验脚本（新增，未跟踪，建议入库以保可复现）
git add scripts/p1_sprint/

# C. 本轮论文深改（已跟踪文件的修改，W1-W4 写作修复 + 口径统一）
git add docs/paper/latex/highlights.tex \
        docs/paper/latex/main.tex \
        docs/paper/latex/refs.bib \
        docs/paper/latex/supplementary.tex \
        docs/paper/latex/sections/01-introduction.tex \
        docs/paper/latex/sections/02-related-work.tex \
        docs/paper/latex/sections/04-experiments.tex \
        docs/paper/latex/sections/05-ablation-analysis.tex \
        docs/paper/latex/sections/06-conclusion-limitations.tex
```

依据：`git status --porcelain`（只读）显示上述 9 个 latex 文件为 `M`（已修改、已跟踪），`reports/p1_sprint_evidence/` 与 `scripts/p1_sprint/` 为 `??`（未跟踪）。`runs/`、`data/`、`external/`、`*.pt` 均被 .gitignore 忽略，**不要**加入。

## 四、Open 项 / 备注

1. ~~**frozen_backbone_contrast 状态矛盾需人工确认**~~ ✅ **已关闭（2026-11-16 终审）**：`runs/p1_sprint/autopilot.log`（cycle 8，00:53）标记 `FAILED-FINAL（重试 99 次）`，但 `frozen_backbone_contrast/summary.json`（23:51，mode=full）三臂数据完整，且与收官口径（frozen 68.6%/176.9s、fullft 98.2%/342.3s、scratch 97.7%/365.7s）一致。**以 summary.json 为准，log 状态为调度器陈旧快照**（重试期间最终成功落盘 summary，调度器未回写状态）。已按 summary.json 归档；queue 状态文件（`runs/p1_sprint/queue/*.status.json`）为失败/运行中的陈旧快照，未归档。
2. ~~**E1（60→49 ten-merge 零重训上界）**~~ ✅ **已完成**（2026-11-16 02:04 归档 `zerobound_6049_summary.json`，源 `runs/p1_sprint/zerobound_6049/summary.json`（01:13 落盘），`cmp` 逐字节一致；脚本 `scripts/p1_sprint/zerobound_6049.py` 01:01 先于运行）。
3. ~~**E2（1,100 clips 对照预算下的语义重建臂）与 E3（合成层 split/addition 迁移）**~~ ✅ **已完成**（2026-11-16 02:04 归档 `semantic_rebuild_1100_summary.json` 与 `split_addition_episode_summary.json`，源分别为 `runs/p1_sprint/semantic_rebuild_1100/summary.json`（02:02）与 `runs/p1_sprint/split_addition_episode/summary.json`（01:50），`cmp` 逐字节一致）。
4. **未归档的周边产物**（仍在 `runs/p1_sprint/`，按需可补）：7 个 `*/summary_smoke.json`、`queue/` 下 7 个 `*.status.json` + `launcher.err.log`/`launcher.out.log`、`autopilot_state.json`、`frozen_backbone_contrast/trainer_artifacts/history.json`、E1–E3 运行日志（`zerobound_6049.log` 919B、`semantic_rebuild_1100.log` 1,458B、`split_addition_episode.log` 929B；三者 `.err` 均为 0 字节）、2 个非空 `.err` 文件（`smq_downstream_ablation.log.err` 1,228B、`frozen_backbone_contrast.attempt1.log.err` 1,678B）。
5. **提交纪律**：本目录仅作证据镜像；`runs/p1_sprint/` 原件保持不动；git add/commit 由用户确认后执行。
6. ~~**autopilot 两个日志镜像为归档时点快照（源文件其后有追加）**~~ ✅ **已刷新（2026-11-16 终审，待办 F 提交前）**：已执行 `cp runs/p1_sprint/autopilot.log runs/p1_sprint/autopilot_run.log reports/p1_sprint_evidence/`，`cmp` 复核两镜像与源逐字节一致（现含 cycle 9–12+ 的 E1–E3 收割记录至 02:23）；4 份 summary.json（zerobound_6049 / semantic_rebuild_1100 / split_addition_episode / k_ladder）亦复核与源逐字节一致。
7. **E3b k 阶梯 summary 补充归档（2026-11-16 统稿）**：`k_ladder_summary.json`（源 `runs/p1_sprint/k_ladder/summary.json`，03:10 落盘）由统稿环节归档，`cmp` 逐字节一致；正文载体为 04-experiments.tex 的 Reinforcement E3/E3b 段 + tab:kladder，协议全文在 supplementary.tex app:kladder。E3b 的两份运行日志（`k_ladder.log`、`k_ladder_s43_44.log`）与 E1–E3 日志同例留在 `runs/p1_sprint/` 未镜像。

8. **真实 tier 冻结对照归档（2026-11-16 终审回修）**：`frozen_real_tier/` 3 份 summary 已 `cmp` 逐字节校验后归档；对应驱动脚本 `scripts/p1_sprint/frozen_ntu120.py`、`frozen_ucf101.py`、`frozen_panaf500.py` 随本轮证据提交入库。
9. **k 阶梯镜像刷新（2026-11-16 终审 P1 收口）**：终审发现 `k_ladder_summary.json` 为补测前旧快照（03:10:01：k≤50 单种子 42、边际 50.69/34.83/32.07/23.45pp、decoupled 0.4517/0.6103/0.6379），与正文引用的 canonical 版（17:52:40：全档位三种子、边际 50.57/36.78/30.80/23.45pp、decoupled 0.4529/0.5908/0.6506）不一致。已重新覆盖为 canonical 版并 `cmp` 逐字节一致；补测会话原件 `summary_s43_44_k102550.json` 同批归档为 `k_ladder_summary_s43_44_k102550.json`。旧快照的中间状态仍可溯源于 `runs/p1_sprint/k_ladder/summary_pre_s43_44_k102550.json`（未镜像）。

10. **头对头鲁棒性矩阵归档（2026-11-17 凌晨，PSD-H2H-MATRIX）**：`h2h_matrix/` 子目录新增 3 份产物——`summary.json`（20/20 格全完成、0 失败，doneMarker=H2H_MATRIX_DONE：A 轴 init 种子 5×2 基线、B 轴子集重抽 s43/s44×3 方法、C 轴预算 5%/20%×3 方法；每格含 per-seed/池诊断/wall-clock）、`analysis.json`（Welch/Mann-Whitney 检验、子集配对差、预算曲线汇总）、`linear_budget_curve.json`（PSD 线性头 5/10/20% 确定性读数，10% 处复现 E9 66.05% 断言通过）。三份均 `cmp` 与源 `runs/p1_sprint/h2h_matrix/` 逐字节一致。正文载体：04-experiments.tex tab:h2h（升级为 mean±std + 5%/20% 预算列 + 显著性）与头对头段落、sec:implementation 追加 post-hoc 双检验族披露、06 节 L13 口径更新；协议全文 supplementary.tex app:h2hmatrix。驱动脚本 `scripts/p1_sprint/h2h_matrix.py`（v2 子进程隔离版，commit f53e7c6）、`analyze_h2h.py`、`linear_budget_curve.py` 已入库。关键数字：PSD 67.53%±0.24（E9 十种子）vs AimCLR 66.74%±0.05（5 init 种子）→ +0.79pp（Welch t=10.1, p=1.1e-6, Holm 双检验族 p=2.3e-6）；vs ST-GCN 64.84%±2.18 → +2.69pp（Welch p=0.051 不显著，MWU p=6.7e-4）；子集重抽 s43/s44 PSD 74.11±0.22 / 74.20±0.08 全档位序保持；预算 5%/20% PSD 74.20±0.05 / 74.08±0.15 vs AimCLR 64.12/69.62、ST-GCN 52.88/72.83。

11. **无标签物理层协议回归归档（2026-11-18，NTU120/UCF101/PanAf500 三 tier）**：`labelfree_tiers/` 子目录 3 份 summary——`ntu120_labelfree_summary.json`、`ucf101_labelfree_summary.json`、`panaf500_labelfree_summary.json`，均 `cmp` 与源 `runs/p1_sprint/labelfree_<tier>/summary.json` 逐字节一致。关键数字（预注册 from-scratch ST-GCN pretext 150ep 无标签 + 同三臂 10 seeds 42–51）：NTU120 ref 58.37%/线性 44.18%/管线 43.75±1.16 → 74.96% FAILS；UCF101 ref 19.87%/线性 9.94%/管线 10.15±2.04 → 51.09% FAILS；PanAf500 ref 64.0%/线性 60.0%/管线 52.8±9.6 → 82.5% FAILS（interim 读数 85.8% PARTIAL 并排保留于 prior_readout）。四 tier 无标签梯度单调：90.7→82.5→74.96→51.09（ref 74.45→64.0→58.37→19.87）。PanAf PSD24→NTU25 槽位映射（前肢→臂槽/后肢→腿槽/躯干合成同 COCO 规则/16–24 死位）见 `export_tier_feeder.py` 模块注释与 supplementary app:frozenreal。驱动脚本 `export_tier_feeder.py`（panaf 分支本轮扩展）、`labelfree_tier_replication.py`、`chain_panaf500_sliced.sh`（6×25ep 分段、段间进程退出）已入库。

12. **E10 域适配三线归档（2026-11-19，NTU120 打样，"让冻结编码器动起来"）**：`adaptation/ntu120/` 6 份产物——`E10_SUMMARY_ntu120.md`（总表）、`summary.json`（T1 TTA）、`summary_lora_samedomain.json`（T2）、`summary_zeroshot.json`（T2b 零样本，由完整 10 种子日志转录重建——汇总写入曾因相对路径 bug 崩溃，bug 已修，测量值逐字转录自 `adapt_loraxfer_zeroshot_run.log`）、`summary_lora_xfer.json`（T2b 跨域 LoRA）、`summary.json`（T3 SupCon 回灌）。**全部 FAILS，无一越过 85% 线**：基线 74.96% / T1 TTA（TENT，BN 仿射，eval 口径）74.08% / T2 同域 LoRA 续训 30ep 74.99%（与基线持平——同数据饱和续训无效）/ T2b 零样本 NTU60→NTU120 69.77%（跨域直接迁移更差）/ T2b 跨域 LoRA 71.98%（full_ref 0.5966 与 10% 线性 0.4527 均为全场最高——绝对可分性提升，但 PSD 语义层未能收割该增益，保留率反降）/ T3 伪标签回灌 SupCon 74.98%（持平）。**科学结论：零额外标注的编码器适配（测试时熵最小化/低秩无标签续训/跨域迁移/伪标签回灌）均不能桥接弱参照域鸿沟——边界是结构性的（标注质量门控物理层质量），与 k-ladder"标签量驱动编码器上限"结论互证**。诊断副产品：① TENT 塌缩根因是 BN train/eval 口径差（批统计使 44% 头掉到 27%），非适配损伤；② SupCon 损失 -inf×0=NaN 陷阱（torch.where 掩码修复）；③ 多进程 OMP 24 线程×4 进程超订阅致假死——线程上限 6 修复。驱动脚本（均入库）：`adapt_common.py`（LoRA 注入 B=0 安全初始化/统一尺复用/supcon_loss）、`tta_tent_tier.py`、`tta_grid_probe.py`、`lora_continue_tier.py`（同域/跨域/零样本三模式）、`supcon_feedback_tier.py`、`eval_adapted_arms.py`、`collect_e10.py`；`labelfree_tier_replication.py` 重构出 `run_arms()` 统一测量函数（行为保持，冒烟复现 0.5837/0.4418）。适配 ckpt：`runs/p1_sprint/adapt_lora_xfer_ntu120/lora_epoch30_model.pt`（跨域，NTU60 phaseB epoch300 为源）。CVIU 投稿包不受影响（e59fd06 锁定）。

13. **E10-T4 预算再分配轮（2026-11-19，用户对 T4 方案批注"执行"；NTU120）**：`adaptation/ntu120/` 新增 4 份——`summary_ref5.json`（冻结+5% 语义）、`summary_split5.json`（5% 锚 LoRA-CE 微调编码器 + 同锚 5% 语义）、`summary_same10.json`（同一 10% 锚既微调编码器又跑语义）、`E10_SUMMARY_ntu120.md`（总表更新）。**逐种子耦合设计（预算诚实）**：每种子用该种子自己的锚集微调（LoRA r=8 + 新线性头，CE+余弦蒸馏锚，BN running stats 冻结保持口径一致），重导特征后全量线性 ref + 同锚 (b) 语义臂。**结果：E10 系列首个正效应**——same10 retention **78.87%**（基线 74.96%，**+3.91pp**，10 种子全部改善，b 43.75→47.10）；split5 **67.51%** vs ref5 64.61%（**+2.90pp**，5% 预算下复现）。机制与 k-ladder 互证：**标签必须塑形编码器**——同一批锚点"既喂语义层又喂物理层"胜过全给语义层，零额外标注成本。仍未过 85% 线（差 6.1pp），但梯度真实、方向明确。驱动脚本 `scripts/p1_sprint/t4_budget_split_tier.py` 入库；烟测曾断言拦截 pkl 标签序误用（pkl=(names,labels)，取 [1]）。后续候选（未执行）：微调超参扫描、预算-分配率曲线、跨 tier 铺开。

14. **E11 扫参+预算曲线+跨 tier 铺开轮（2026-11-19，用户裁决"扫参+跨 tier 铺开、深挖不设限"；因"不要挂长后台占用内存"约束改为前台分片+逐种子断点续跑）**：`adaptation/` 新增 `E11_SUMMARY.md` 总表、`ntu120/summary_sweep_*.json`（d0/lr1e2/lr2e2/lr4e2/ep200/lr1e2_r16 六配置）、`summary_ref15.json`+`summary_dual15.json`（15% 预算点）、`ucf101/summary_same10.json`、`panaf500/summary_same10.json`（修复后重跑）。**核心发现**：①**学习率是唯一显著超参**——lr1e-3→78.87%、1e-2→82.13%、2e-2→82.73%（NTU120 同 10% 双用），4e-2 语义臂校准崩溃（tau=0），epochs/rank/λ 均不敏感（±0.5pp）；②**预算-分配率曲线单调**：5/10/15% 冻结=64.61/74.96/79.04%，双用=67.51/78.87/**83.60%**——双用优势随预算扩大（+2.90/+3.91/+4.56pp）；③**跨 tier：UCF101 51.09%→71.21%（+20.12pp，最弱域受益最大）**，PanAf500 82.5%→82.5%（40 锚点数据饥饿+25 clip val 量化，首批修复前微调零迭代、修复后特征漂移仍不翻转离散管线——可复现性副产品：整管线逐 bit 确定性）。NTU120 最优 82.73% 距 PARTIAL 线 2.27pp。工程：`t4_budget_split_tier.py` 增加 --tag/--seed-list/逐种子断点续跑/小锚集 eff_batch 修复；前台分片模式=每片单种子（timeout 560s）+ OMP 12 线程。commit 见 git log；CVIU 包不受影响。

15. **E12 PanAf 数据饥饿干预轮（2026-11-19，用户批注"继续冲击！！！"）**：针对 PanAf 双用无变化的根因（40 锚点喂不饱微调），t4 脚本新增 `--aug-views`（锚点经 Feeder_single shear0.5/pad6 增广采样，每 epoch n×views 个增广视图）。强压配置（aug8+λ0 / aug8+λ1.0）均触发语义臂校准门禁崩溃（tau=0 硬报错——PSD 门禁拒绝退化校准，响亮失败不产垃圾）；**温和配置 aug4+lr1e-2+λ1.0+ep50 打通：PanAf500 10 seeds retention 82.5% FAILS → 92.21% CONFIRMS（+9.71pp，全项目首个 CONFIRMS）**，mean_b 0.528→0.568，6/10 种子改善、2 平 2 降（25 clip val 量化 4% 步长；符号检验不显著，判定翻转部分来自 ref 0.64→0.616 漂移——已带量化/显著性保留意见如实记录）。副产品：lr4e-2（NTU120）与 PanAf 强压配置证明**过强编码器适配会触发语义层校准门禁**——适配强度存在上界。证据 `adaptation/panaf500/summary_aug4.json`+`E11_SUMMARY.md`（跨 tier 表更新为各 tier 最优配置）。脚本 `t4_budget_split_tier.py` +--aug-views/--tag。

16. **E13 增广敏感性补全轮（2026-11-19，用户"持续推进不要停"）**：①NTU120 增广配置——aug4+lr2e-2 → **83.41%**（NTU120 最优，距 PARTIAL 1.59pp），aug4+lr1e-2 → 82.06%（mean_b 0.5164/mean_ref 0.6293 均最高但保留率低于 lr2e-2——ref 升幅大于 b）；②UCF101 aug4+lr1e-2 → 68.31%（低于 lr2e-2 无增广的 71.21%——lr2e-2 仍是 UCF 最优）；③PanAf aug4+lr2e-2 触发校准门禁（tau=0 崩溃，lr2e-2 在 40 锚点上过强）——**各 tier 最优配置不同：NTU120=lr2e-2+aug4、PanAf=lr1e-2+aug4、UCF=lr2e-2**，与锚点质量/数量一致（锚多可承受强适配）。证据镜像 ntu120/ucf101 增配 aug4 文件；E11_SUMMARY 三表更新。
