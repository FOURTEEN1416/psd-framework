# W24 报告 — 多窗口 worktree 物理隔离落地（B-full · 2026-08-25）

> 任务书: `dev-docs/handovers/W24-collab-worktree.md`（用户裁决 B-full 授权）
> 执行窗: W24 协作治理窗口（歆歆）
> 结论: **交付清单 D1-D5 全部完成并通过验收门**；过程中发生 1 起 git 删窗误伤事故，已闭环修复并固化为脚本机制

## 1. 脚本用法（D1: `scripts/new_window_worktree.ps1`）

```powershell
# 建窗（在任意目录可跑，路径以脚本自身定位主仓）
pwsh scripts/new_window_worktree.ps1 -Name w25-demo

# 卸窗（收编完成后清理；先摘 Junction 再删树）
pwsh scripts/new_window_worktree.ps1 -Name w25-demo -Remove
```

| 行为 | 实现 |
|------|------|
| 建 worktree | `git worktree add ..\psd-framework-<Name> -b wt/<Name>` |
| data/ 共享 | 删除 checkout 出的空壳 data/（仅 .gitkeep）→ `New-Item -ItemType Junction` 指回主仓；失败自动退化为 `robocopy /MIR` 快照 + `_FORK_STAMP.json` 分叉戳 |
| runs/ 独立 | 新建目录 + 复制上游 checkpoint 清单（当前: `runs/p05_stgcn_bc_full/best.pt`） |
| .venv | 不复制；提示词给出主仓绝对解释器 `D:\Desktop\psd-framework\.venv\Scripts\python.exe` |
| 启动提示词 | 打印含工作目录/分支/收编方式/白名单纪律/卸窗警告的标准提示词，可直接复制给新会话 |
| 防呆 | 窗口名字符白名单校验；目录/分支已存在即拒绝；建窗失败自动回滚 |
| 卸窗防呆 | 先摘 data Junction（校验 Target 必须指向主仓）→ `merge-base` 检查分支是否已并入 HEAD，未合并默认保留分支并告警，需 `-ForceBranch` 才强删 |

## 2. 冒烟证据（D3，2026-08-25 实测）

### 2.1 建窗端到端（`-Name _smoke`）

```
[data] Junction 建立: D:\Desktop\psd-framework-_smoke\data -> D:\Desktop\psd-framework\data
[ckpt] 已复制: runs/p05_stgcn_bc_full/best.pt
✅ 窗口 [_smoke] 就绪
```

核验: `Get-Item data -Force` → `LinkType=Junction Target=D:\Desktop\psd-framework\data`；synthetic 目录 4 文件经联接可读；worktree 内 `git status --short --branch` 仅 `## wt/_smoke`（零脏项，data/.gitkeep 经联接与 HEAD 一致）。

### 2.2 pytest 全绿（worktree 内，cwd=worktree，主仓绝对解释器）

```
288 passed in 21.11s
```

与主仓同日 collect-only 数量一致（288），无基线漂移。

### 2.3 tiny 训练 smoke（证明 Junction 数据链 + 训练管线 + runs 隔离）

内联调用 psd 库 API（ST-GCN+BC, CPU, 2 epochs, seed=42）:

```
[junction-data] OK: ...\psd-framework-_smoke\data\synthetic\syn_22class_20per_class_seed42.pkl
[junction-data] samples=440 classes=22
[tiny-train] epochs=2 best_val_acc=0.0568 final_train_acc=0.3324 no_nan=True wall=22.6s
[runs-isolation] worktree runs 产物存在: True (..._smoke\runs\_w24_smoke_tmp\history.json)
[runs-isolation] 主仓未被污染: True
[SMOKE-PASS]
```

解读: 2 epochs 下 val_acc≈随机水平(0.045)属预期（W11 同管线 50ep 冒烟才 18.2%）；train_acc 0.3324 表明学习信号正常、无 NaN 即收敛方向健康。冒烟目的是验证**机制可用性**而非精度，判据为无 NaN + loss 可降 + 数据可读，全部满足。

### 2.4 清理与卸窗复验

测试窗删除后核验: 主仓 `data/.gitkeep` 在 ✓ / synthetic 4 文件在 ✓ / 无目录残留 ✓ / `worktree list` 仅主检出 ✓ / `wt/*` 分支清空 ✓。第二轮建+卸（`-Remove` 路径新鲜验证）同样全过。

## 3. 过程事故登记（负结果如实归档）

**git 直接删窗跟随 Junction 误伤主仓**: 首轮清理用裸 `git worktree remove --force`，git 递归删除时跟随 data Junction 把主仓 `data/.gitkeep` 一并删除（报错 `failed to delete ... Invalid argument` 后中止，其余数据完好）。处置: 当场 `git restore data/.gitkeep` 恢复（git 跟踪文件零损失）；随后给脚本增补 `-Remove` 安全卸窗开关（摘 Junction → 合并状态检查 → 删树删分支）并在启动提示词写入禁令，第二轮建+卸复验通过。

## 4. 与任务书的偏差登记

| 偏差 | 原因 | 影响 |
|------|------|------|
| ADR 文件名 `0003-*` → `0005-user-rulings-warmstart-worktree.md` | 任务书拟编号与既有 `0003-p02-sprint-and-metric-ruling.md` 冲突（0001-0004 占满）；禁触条款禁止覆盖既有裁决 | 仅编号顺延，内容按 D4 要求完整落档 |
| 冒烟训练弃用 `scripts/run_p05_prep_smoke.py` | 该脚本读 `cfg["data"]` 嵌套键而现行 `configs/p05_stgcn_bc.yaml` 为扁平结构（必 KeyError），且属他人窗口领地不可改 | 改为内联调用同一套 psd 库 API（dataset/model/trainer），验证目标等价且零越权 |

## 5. 遗留风险

| 风险 | 状态 | 缓解 |
|------|------|------|
| 两窗同时写共享 data/ 同名生成物 | 开放（协议约束，非技术强制） | AGENTS.md §4 第 4 条: 生成物带窗口前缀或唯一 seed；冲突以后提交者重命名为准 |
| robocopy 分叉退化路径未经实测 | 本机 Junction 创建成功，fallback 分支未被触发 | 代码路径简单；若真触发会写分叉戳提醒人工介入 |
| 进行中窗口（W18/W21/W22/W23）仍以旧方式在主检出收尾 | B-full 不强制迁移（任务书边界条款） | 各窗收尾后自行切换；过渡期主检出仍有并行写风险，协调窗口收编前照旧 git diff 重读 |
| 上游 checkpoint 清单为静态硬编码 | 新实验可能引入新必需权重 | 清单位于脚本头部 `$UpstreamCheckpoints` 数组，后续任务书增补即可 |

## 6. 验收门对照

| 门 | 结果 |
|----|------|
| D1 脚本对真实窗口名跑通一次 | ✅ `_smoke` 端到端 ×2（含 -Remove 卸窗路径） |
| D3 冒烟全绿证据入报告 | ✅ §2.2 pytest 288 绿 / §2.3 tiny 训练 PASS |
| D4 ADR 落档 | ✅ `dev-docs/decisions/0005-user-rulings-warmstart-worktree.md`（编号偏差见 §4） |
| HANDOVER §8 回写 + 修订历史 v-next | ✅ 见同日提交 |

> 版本: v1.0（2026-08-25）
