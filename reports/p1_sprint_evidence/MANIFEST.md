# P1 Sprint 证据归档清单（reports/p1_sprint_evidence/）

> 归档时间：2026-11-16 · 操作：证据归档员（只复制/新建，未删除任何文件，未执行 git add/commit）
> 来源：`runs/p1_sprint/`（该目录被 `.gitignore:19 /runs/*` 忽略，故将可提交证据镜像至本目录）
> 完整性校验：全部 19 个文件已用 `cmp` 与源文件逐字节比对，**全部一致**（E1–E3 三份 summary 于 2026-11-16 02:04 补充归档；E3b k 阶梯 summary 于同日统稿时补充归档；本清单更新时已再次 `cmp` 复核，仍与 `runs/p1_sprint/*/summary.json` 逐字节一致）。
> **2026-11-16 终审回修补充归档（真实 tier 冻结编码器对照）**：`frozen_real_tier/` 子目录新增 3 份 summary（ntu120/ucf101/panaf500，源 `runs/p1_sprint/{ntu120,ucf101,panaf500}/summary.json`，13:04–14:01 执行），逐份 `cmp` 逐字节一致。

## 一、归档文件（19 个 + 本清单 MANIFEST.md）

### 实验结果 summary.json（8 个，逐字节校验通过）

| 归档路径 | 源路径 | 关键内容 |
|---|---|---|
| `merge_rename_upper/summary.json` | `runs/p1_sprint/merge_rename_upper/summary.json` | merge/rename 零重训上界（ntu60→ntu120 10 类，seed 42，预算 10%=4009 clips）：rename-oracle 上界 top1 **0.7676**，argmax-vote 下界 0.7651，目标全监督参照 0.715，retention 1.0735 |
| `standard_selftrain_baseline/summary.json` | `runs/p1_sprint/standard_selftrain_baseline/summary.json` | 标准自训练基线（τ=0.95，10 iters）：std-ST top1 0.6639 vs 参照 linear10 0.6605，Δ=+0.0034；含 τ∈{0.8,0.9} 敏感性；耗时 5899.9s |
| `smq_downstream_ablation/summary.json` | `runs/p1_sprint/smq_downstream_ablation/summary.json` | SMQ 下游消融（225 clips，7 伪标签类）：with_smq 0.4595 vs without_smq 0.4595，**Δ=0.0**；耗时 48.1s |
| `frozen_backbone_contrast/summary.json` | `runs/p1_sprint/frozen_backbone_contrast/summary.json` | 冻结三臂对照（50 clips/class，seed 42，100 epochs）：**frozen 0.6864 / 176.9s，fullft 0.9818 / 342.3s，scratch 0.9773 / 365.7s**（frozen_verified=true，可训练参数 6,423） |
| `zerobound_6049_summary.json` | `runs/p1_sprint/zerobound_6049/summary.json` | E1：TRANS-002 同 episode（60→49 ten-merge）零重训上界重测（seed 42，10% 标注头预算 4,009 clips，纯推理 112.5s，零重训零新标注）：rename-oracle 上界 **0.6804**，argmax-vote 下界 0.6797，源 60 类头 0.6605，目标 49 类全监督参照 0.6756（retention 1.007） |
| `semantic_rebuild_1100_summary.json` | `runs/p1_sprint/semantic_rebuild_1100/summary.json` | E2：TRANS-002 三臂 @ 1,100-clip 对照预算（seeds 42/43/44，80 epochs，预算抽取 seed 42）：C′ 0.4493 / 180.0s，D′ **0.6016 / 6.8s**，M 0.4820 / 10.2s；D′−C′ = +15.23pp，成本比 **26.47×** |
| `split_addition_episode_summary.json` | `runs/p1_sprint/split_addition_episode/summary.json` | E3：合成层 22→29 split/addition 迁移 episode（3 拆分 + 4 新增，seeds 42/43/44，50 clips/class，60 epochs）：remap 0.4299 → decoupled(k=10) 0.4575 → frozen head 0.6805 → coupled **0.9713**；新类精度阶梯 + 中位成本比 **3.02×**（verdict PARTIAL） |
| `k_ladder_summary.json` | `runs/p1_sprint/k_ladder/summary.json` | E3b：同 episode 定义标注预算阶梯 k∈{10,25,50,100}（episode seed 42；k≤50 单种子 42，k=100 三种子 42/43/44；池扩到 100/类）：decoupled 0.4517/0.6103/0.6379/**0.7241**（k=100 ±0.0），coupled 0.9586±0.0049（k 无关参照），frozen head 0.7172±0.0049；coupled−decoupled 边际 50.69/34.83/32.07/**23.45pp** 全部在 ±2.3pp 平价带外 → **k\*=None（verdict=no_parity_within_tested_range）**；k=10 档 100 vs 1000 新标注、92.1s vs 234.0s（seed 42） |

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
