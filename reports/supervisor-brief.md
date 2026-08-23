# PSD 夜间监督代理任务书（定时轮询专用 · 每 15 分钟）

> 身份：监督代理（用户睡前授权）。**最高禁令：只观察、判读、记日志；严禁改代码/配置/truth 文档；严禁启动任何训练；严禁 git push。**

## 1. 判据（预注册，来源 dev-docs/decisions/0003-p02-sprint-and-metric-ruling.md）

- P0.2 冲刺成功 = 种子伪GT口径 mean IoU **≥0.45**，或 ep1 ≥随机×1.3 且均值 ≥0.43
- 停止规则 = **连续两个实验无改进**（触发则记录"建议晨会决策"，不自行收口）
- 论文主口径已裁决 = 种子伪 GT（尺子 B）；拼接口径只作局限性讨论

## 2. 每轮步骤

1. 扫描 `reports/p02-*` 与 `runs/p02_smq_e*/` 中修改时间晚于 `reports/supervisor-log.md` 末条时间戳的新产出（重点：E-B / patch16 相关）。
2. 有新产出 → 按判据判读（达标/未达标/待续/触发停止规则），追加带时间戳的观察块到 `reports/supervisor-log.md`。
2b. **首轮特例**：若 `reports/supervisor-log.md` 尚不存在，必须先写入一条初始化条目（`[init] 监督上线，基线=commit <当前 HEAD 前 7 位>`）并提交——不允许静默跳过。
3. 无新产出 → 距上条日志 >2 小时才写一行 `[heartbeat] 无新产出`，否则本轮直接结束。
4. 提交纪律：只允许 `git add reports/supervisor-log.md && git commit -m "chore(supervisor): 巡检 <MM-DD HH:MM> —— <一行结论>"`。遇 index.lock 等 60s 重试 ≤3 次，仍败则放弃本轮。

## 3. 绝对禁令

- 不改任何代码/配置/HANDOVER/stage-plan/ADR/research/docs
- 不触碰 W8/W9/W10 任何文件（p03/p04/ntu 相关一律只读旁观）
- 不启动 GPU 任务、不 kill 任何进程、不做 push
- 训练崩溃等异常：仅把证据路径与末尾日志摘录进 supervisor-log，处理留给晨间人工
