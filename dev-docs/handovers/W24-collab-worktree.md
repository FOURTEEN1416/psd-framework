# W24 任务书 — 多窗口 worktree 物理隔离落地（用户裁决 B-full · 2026-08-24）

> 状态: ✅ 完成（2026-08-25，报告 `reports/w24-worktree-governance-2026-08-25.md`；ADR 因编号冲突落档为 0005）
> 背景: W14 登记并行窗口 4 次覆盖已提交文件的事故（`reports/w14-p05-efficiency` 报告 §6，正确路径 `reports/w14-p05-al-efficiency-2026-08-24.md`）；同期 rescue-plan 登记了 W16 映射证伪等协作事故。根因 = 多窗口共用同一工作树，旧缓冲/误操作直接覆盖他人产物
> 裁决依据: 用户 2026-08-24 于歆歆协调会话拍板 **B-full（每窗口 git worktree 物理隔离）**，授权以本册为准（否决 B-lite 仅纪律方案）

## 0. 目标协议（一句话）

新窗口一律在自己的 worktree 中开工，主检出目录（`D:\Desktop\psd-framework`）只保留协调合并用途；跨窗产物经 git merge 收编。

## 1. 交付清单

### D1 一键建窗脚本 `scripts/new_window_worktree.ps1`
- 参数 `-Name <窗口名>`，行为：
  1. `git worktree add ..\psd-framework-<Name> -b wt/<Name>`（分支命名 `wt/<Name>`）
  2. **data/ 目录联接**：`New-Item -ItemType Junction` 指回主仓 data/（gitignore 生成物不随 worktree 走；Junction 无需管理员权限）
  3. **runs/ 独立新建** + 上游 checkpoint 按需复制清单（当前已知必需：`runs/p05_stgcn_bc_full/best.pt`；后续任务书可增补）
  4. `.venv` 不复制——文档写明用主仓绝对解释器：`D:\Desktop\psd-framework\.venv\Scripts\python.exe`（cwd 在 worktree 内即可正确 import psd）
  5. 打印该窗口的标准启动提示词与白名单提醒
### D2 AGENTS.md「并行纪律」节增补 worktree 协议
- 新窗口必须 worktree 开工；主检出只做 merge；提交仍在各自 wt 分支、收编走显式 merge；白名单制度在 worktree 内照旧执行
### D3 冒烟验证（不可跳过）
- 建 `wt/_smoke` 测试窗 → 在其中跑 `pytest psd -q` 全绿 → 跑一个 tiny 训练 smoke 证明 Junction 数据可读 → 删除测试 worktree 与分支
### D4 用户裁决落档
- `dev-docs/decisions/0003-user-rulings-warmstart-worktree.md`：登记 A2（warm-start 协议采纳）+ B-full（worktree 隔离采纳）两项裁决原文与生效范围（本册即授权依据）
### D5 报告
- `reports/w24-worktree-governance-2026-08-XX.md`：脚本用法、冒烟证据、遗留风险

## 2. 边界与禁触

- **不迁移进行中窗口**：W18 巡检 / W21 / W22 按原方式在主检出收尾后自行切换；本窗只交付机制
- 禁触：`docs/paper/**`、`*ntu*`、`external/**`；dev-docs/decisions/** 仅允许 D4 新增文件，不改既有裁决
- Conventional Commits 中文提交

## 3. 风险预案

| 风险 | 预案 |
|------|------|
| Junction 创建失败（权限/杀软拦截） | 退化为 `robocopy //MIR data <worktree>\data` 一次性复制 + 数据版本戳登记（注明非共享，存在分叉可能） |
| worktree 内 pytest 失败 | 排查 REPO_ROOT/cwd 假设（脚本应以 `__file__` 为基准，理论无碍）；失败留证上报，不得改库代码迁就 |
| 两窗同时写 data/ 同名生成物 | data 共享即同盘写入——约定生成物带窗口前缀或唯一 seed，冲突时以后提交者重命名为准 |

## 4. 验收门

- [x] D1 脚本对真实窗口名跑通一次（_smoke 端到端 ×2，含 -Remove 卸窗路径）
- [x] D3 冒烟全绿证据入报告（pytest 288 绿 + tiny 训练 + Junction 链验证）
- [x] D4 ADR 落档（编号冲突顺延: 0005-user-rulings-warmstart-worktree.md）
- [x] HANDOVER §8 本行状态回写 + 修订历史 v2.0 登记

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-24 | 建册（用户裁决 B-full 生效版） |
| v1.1 | 2026-08-25 | W24 执行完毕: D1-D5 全交付，验收门全勾；偏差登记 ADR 编号顺延 0005、冒烟改内联 API 调用 |
