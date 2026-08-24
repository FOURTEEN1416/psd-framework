# W18 任务书 — GPU 排程看护（排队任务自动接力）

> 窗口: W18（全新独立窗口）
> 日期: 2026-08-24 | 编制: 协调者歆歆
> 背景: RTX 5060 8GB 被 NTU Phase B 长训独占（数天级）；两个 full-budget 任务排队等待

## 1. 排队任务清单（严格串行，禁止并发）

| 序 | 任务 | 启动命令 | 前置条件 |
|----|------|---------|---------|
| Q1 | AL full-budget 复跑 | `.venv\Scripts\python.exe scripts\run_p05_al_efficiency.py --config configs\p05_al_full.yaml --fresh` | config 内 RUNDATE 改为当日 |
| Q2 | C1 解耦成本 full 档 | `.venv\Scripts\python.exe scripts\run_c1_decouple.py --n-per-class 100` + 完整 epochs 参数 | **W19 已提交可运行版本且 TDD 全绿**（若 W19 未完成则跳过并登记） |

## 2. 巡检协议

每 30 分钟一轮：
```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "ntu|p05|c1" } | Select-Object ProcessId,CommandLine
```

**接力触发条件**（全部满足才启动下一任务）:
1. 显存占用 < 500 MiB
2. 无任何 psd-framework python 训练进程存活
3. 当前无本看护窗口已启动的任务在跑

**启动后监控**: 每 30 分钟查日志尾部；NaN 连续 3 行 → 终止留证（参照 `dev-docs/handovers/W9-ntu-phaseb-schedule.md` 熔断条款）；OOM → 等 10 分钟重试一次，再失败即上报用户。

## 3. 完成动作（每个任务完成后立即执行）

1. 结果 JSON/md 归档确认存在（Q1: `reports/p05-al-efficiency-full-<日期>.json`；Q2: `reports/c1-decouple-cost-*`）
2. Conventional Commits 中文提交证据文件
3. 在 commit message 注明接力触发时间与 GPU 空闲时长
4. 更新本文档 §4 执行记录表

## 4. 执行记录

| 时间 | 事件 | GPU 状态 | 动作 |
|------|------|----------|------|
| （待填） | | | |

## 5. 领地边界

**可写**: `reports/` 下两个任务的产物、本文档 §4、`runs/` 对应输出目录
**禁触**: 一切代码文件、`docs/paper/**`、`dev-docs/decisions/**`、`*ntu*` 文件本体（只监控不干预 NTU 进程——它不是你的猎物）

## 6. 特殊情况

- 若巡检期间发现非 NTU/非排队的陌生训练进程：**不杀不动**，登记后上报用户裁决
- 若 NTU 长训中途自然结束：等 10 分钟确认无重启后按触发条件接力 Q1
- 用户手动跑实验导致条件永不满足：耐心等，禁止抢卡

## 修订历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-24 | 建册：两任务串行接力 + NaN/OOM 熔断条款 |
