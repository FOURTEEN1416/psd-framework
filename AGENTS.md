# AGENTS.md — PSD-Framework Agent Constitution

> 状态: v1.1（2026-09-03 补行为准则节）
> 方法论继承自 k9-training-system AGENTS.md（sliver-vibe-coding 框架），此处只保留本仓库必需条款

## 0. 行为准则（个人级，全局优先）

个人行为规则（五条铁律详版 / 决策人话条款 / 防乐观完成 / 用户偏好）由 **全局 `C:\Users\FOUR\.zcode\AGENTS.md` v2.0** 提供，本文件不重复。冲突时：**本仓硬规则（下节）在工程口径上优先于全局**——特别是硬规则 1（GitHub-First 调研零容忍、禁 WebSearch）覆盖全局 Research-First 的工具白名单；全局的"决策人话条款/用户偏好"在本仓同样强制生效。

## 1. 项目身份

学术研究仓库：物理-语义解耦低资源动物行为识别框架（论文 + 基座方法论）。
产品功能一律不做（归 k9-training-system）。Truth root: `dev-docs/`。

## 2. 硬规则

1. **GitHub-First 调研（零容忍）**：技术调研完全禁止 WebSearch；必须使用 GitHub 工具链（搜索 → awesome 目录发现 → 逐仓验证活跃度/Issues），违反触发漂移处理
2. **Truth 单一性**：每个概念只有一个 owner 文档，禁止重复 truth
3. **三层指标口径**：合成 / 公开真实 / 真实 K9 分别汇报，禁止混报
4. **无新鲜验证，无完成声明**：所有实验结论必须有当次运行证据 + 报告归档 `reports/`
5. **外部方法仓库**进 `external/`（gitignore）：不修改其内部实现；适配/封装代码写在本仓库 `psd/` 包内
6. **跨仓边界**：只允许文档指针指向 k9-training-system，禁止跨仓 import；代码复用走 `docs/assets-map.md` 显式移植
7. Conventional Commits 中文描述；代码注释语言跟随用户最新消息

## 4. 并行纪律（多窗口 worktree 协议，2026-08-25 增补）

> 背景：W14 曾发生并行窗口共用主检出、4 次覆盖已提交文件的事故；2026-08-24 用户裁决 **B-full（每窗口 git worktree 物理隔离）**，W24 交付机制。本节为所有新窗口的强制协议。

1. **新窗口一律 worktree 开工**：开工前执行 `pwsh scripts/new_window_worktree.ps1 -Name <窗口名>`，此后一切读写只在自己的 `..\psd-framework-<窗口名>` 内进行；分支命名 `wt/<窗口名>`
2. **主检出只做协调合并**：`D:\Desktop\psd-framework` 保留给歆歆协调与 merge 收编用途，不在其中开新的实验窗口；跨窗产物收编一律显式 `git merge --no-ff wt/<窗口名>`
3. **提交仍在各自 wt 分支**：产物落盘即提交；白名单制度在 worktree 内照旧执行——只提交任务书白名单内文件，精确 `git add`，禁用 `git add .`
4. **data/ 共享即同盘写入**：worktree 的 data/ 经 Junction 指向主仓 data/（gitignore 生成物不随 git 走）；各窗口生成物必须带窗口前缀或唯一 seed，冲突时以后提交者重命名为准
5. **runs/ 各窗独立**：上游 checkpoint 由建窗脚本按清单复制；禁止写入他窗 runs 子目录
6. **Python 解释器统一用主仓绝对路径** `.venv` 不复制：`D:\Desktop\psd-framework\.venv\Scripts\python.exe`（cwd 置于本 worktree 内即可正确 import psd）
7. **收编自助**：窗口完成验收自检后运行 `pwsh scripts/window_checkin.ps1 -Name <本窗名> [-Remove] [-Message "移交说明"]` 自助合并——脚本强制执行领地扫描/窗口内测试/master 回归三道门禁，冲突自动中止上报；`-Remove` 卸窗含 runs/data_campaign 数据汇聚检查。禁止绕过脚本直接改 master
8. **跨窗看板**：所有跨窗信息（移交/发现/阻塞/待裁决）必须写入 `dev-docs/board/BOARD.md`（工具 `scripts/window_board.ps1 -Append/-Tail`）；开窗第一件事读看板再读 HANDOVER；协调者监控看板代替逐窗轮询
9. **记忆库双写**：重大发现/裁决/状态变更在写 BOARD 的同时必须调用记忆 MCP（`memory_memory_add`/`bulk_add`，agent_id=shared）入库；开工/验收前用 `memory_memory_context`/`memory_memory_search` 拉取相关记忆——语义召回补 BOARD 的关键词盲区，双信道缺一不可

## 5. Owner Map

| 模块 | 路径 |
|------|------|
| 数据加载 | `psd/data/` |
| 模型（AimCLR / SMQ / TCL / 姚青迁移 / 骨干） | `psd/models/` |
| 训练与评估管线 | `psd/training/` |
| 入口脚本 | `scripts/` |
| 实验配置 | `configs/` |
| 阶段计划 | `dev-docs/stage-plan.md` |
| 决策记录 | `dev-docs/decisions/` |
| 评估报告 | `reports/` |

## 6. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-23 | 建仓初始化，继承上游硬规则（GitHub-First / truth 单一性 / 三层口径） |
| v1.1 | 2026-08-25 | W24 增补 §4 并行纪律（B-full worktree 协议六条），Owner Map 顺延为 §5 |
| v1.2 | 2026-08-25 | §4 增补条款 7-8：收编自助（window_checkin.ps1 三门禁协议）+ 跨窗看板（BOARD.md 强制信道）——消除协调瓶颈与信息不互通 |
| v1.3 | 2026-08-25 | §4 增补条款 9：记忆库双写协议（BOARD 正式信道 + 记忆 MCP 语义召回，agent_id=shared）——补齐跨窗知识管理的语义盲区 |
