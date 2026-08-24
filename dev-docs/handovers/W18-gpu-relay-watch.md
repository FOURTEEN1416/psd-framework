# W18 任务书 — GPU 排程看护（排队任务自动接力）

> 窗口: W18（全新独立窗口）
> 日期: 2026-08-24 | 编制: 协调者歆歆
> 背景: RTX 5060 8GB 被 NTU Phase B 长训独占（数天级）；两个 full-budget 任务排队等待

## 1. 排队任务清单（严格串行，禁止并发）

| 序 | 任务 | 启动命令 | 前置条件 |
|----|------|---------|---------|
| Q1 | AL full-budget 复跑 | `.venv\Scripts\python.exe scripts\run_p05_al_efficiency.py --config configs\p05_al_full.yaml --fresh` | config 内 RUNDATE 改为当日 |
| Q2 | C1 解耦成本 full 档 | `.venv\Scripts\python.exe scripts\run_c1_decouple.py --n-per-class 100` + 完整 epochs 参数 | **W19 已提交可运行版本且 TDD 全绿**（若 W19 未完成则跳过并登记） |
| Q3a | YOLO dog-pose 微调（24 点犬类权重） | `.venv\Scripts\python.exe scripts\train_yolo_dogpose.py --epochs 50 --batch 16`（~1h） | Q2 已完成；`D:\Desktop\datasets\dog-pose` 数据在位（8476 图已预下载核验）；产物 `runs/public_real_yolo_dogpose/train/weights/best.pt` |
| Q3b | AK 犬科视频全量提点 | `.venv\Scripts\python.exe scripts\run_p05_public_real_pipeline.py --stage extract --weights runs/public_real_yolo_dogpose/train/weights/best.pt`（~10min） | Q3a 的 best.pt 存在；172 样本 manifest 在位 |
| Q3c | ST-GCN+BC 公开真实层微调（4 类子集） | 按 `scripts/run_p05_public_real_pipeline.py` 微调 stage 执行（`--stage` 取值以脚本 `--help` 为准；backbone 冻结 + 4 类新 head，init `runs/p05_stgcn_bc_full/best.pt`） | Q3b 提点产物完整过 24 点防呆断言 |

**Q3 特别条款**（用户 2026-08-24 裁决 A 落地）：
1. Q3a→Q3b→Q3c 内部严格串行，任一步失败即停止后续步骤、留证上报；
2. W18 只负责机械执行与 GPU 调度——**科学判读归 W20**：Q3 各步完成后在 commit/执行记录中 @W20 验收（提点质量目检比例、4 类精度数字解读由 W20 出具）；
3. 失败熔断时**禁止自行修改任何代码/配置**（含超参）——那是 W20 的领地；只留证上报等裁决。

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

1. 结果 JSON/md 归档确认存在（Q1: `reports/p05-al-efficiency-full-<日期>.json`；Q2: `reports/c1-decouple-cost-*`；Q3: Q3a 权重文件 + Q3b 提点产物 + Q3c `reports/p05-public-real-partialclass-result-<日期>.json`）
2. Conventional Commits 中文提交证据文件
3. 在 commit message 注明接力触发时间与 GPU 空闲时长
4. 更新本文档 §4 执行记录表

## 4. 执行记录

| 时间 | 事件 | GPU 状态 | 动作 |
|------|------|----------|------|
| 08-24 19:58 | W18 接管：首轮巡检 + 前置核验 | util 99% / 7609 MiB；NTU 双进程 PID 33772(.venv)+35208(Py312) 跑 `run_ntu_phaseb.py`（epoch20 iter6100 loss≈12.8 收敛正常）；⚠️ console_out.log 最后写入 18:40，存在输出滞后观察项——按 §5 只监控不干预 | Q1 前置就绪：`configs/p05_al_full.yaml` RUNDATE→2026-08-24；Q2 前置不满足（W19 任务书 08-24 19:53 刚建册、git 无任何 C1/W19 功能提交），按 §1 条款登记跳过；30 分钟定时巡检注册中 |
| 08-24 20:08 | 例行巡检: NTU 占用中(ntu_proc=2), 继续等待 | 100 %, 7668 MiB | 无条件变化, 不干预 |
| 08-24 20:12 | 例行巡检: NTU 占用中(ntu_proc=2), 继续等待 | 100 %, 7660 MiB | 无条件变化, 不干预（调度器首轮自动运行, LastTaskResult=0） |
| 08-24 20:14 | 看护机制上线：Windows 计划任务 `\OpenCode\w18-gpu-relay-watch` 每 30min 触发 | 同上 | 脚本位于用户级 `~/.config/opencode/jobs/w18-gpu-relay-watch.ps1`(仓库零侵入)；状态机 waiting→q1_running→done/halted_nan/halted_oom；NaN 连续3行/OOM 重试一次熔断已内置；Q1 完成自动归档+中文提交；异常留证本表待裁决 |
| 08-24 20:16 | 用户令复核 → 发现并修复 2 处问题 | 100% / ~7.6GiB | ① Ensure-ConfigDate 正则不含连字符, 跨日无法更新日期文件名——已修 `[0-9A-Z\-]+` 并实测双匹配通过; ② 本表前 3 行因人工编辑错序——已重排时间序。另登记: 工作区存在 W19/W20 未提交产物(run_c1_decouple.py 大改+test_c1_decouple.py+ak_mapping.py), 属并行窗口领地本窗禁触; Q1 完成时提交仅 add 报告/config/本任务书三路径, 不会裹挟他人文件 |
| 08-24 21:01 | 二轮复核验收 → 逐条款对照发现 4 处缺陷, 全部修复并实跑验证 | 100% / 7683 MiB | A 补 §6.1 陌生训练进程检测(初版按命令行匹配误伤 MCP/daemon 常驻服务×6 → 改为 nvidia-smi compute-apps 占卡判定, 21:01 实证误报消除); B 补 §6.2 空闲 10 分钟确认期(idle_since 门禁, 防 NTU OOM 自动重启撞车); C git commit 增加退出码检查, 失败置 q1_commit_failed 留证上报, 杜绝假完成; D 「GPU 空闲时长」语义修正为显存释放→接力的等待时长(relay_wait_min)。20:42 轮曾登记的瞬时 p05/c1 进程(PID 37156/14140)已自行消失, 判定为 W19 窗口测试活动痕迹, 无需处理 |
| 08-24 23:10 | 三轮复核验收 → 刀口转向 Q1 可运行性实证, 发现 2 缺陷 + 1 作用域 bug | 98% / 7657 MiB | ① 计划任务默认拔电即停(StopIfGoingOnBatteries=True)——已改电池双开关=False 并回读确认, 数天级窗口防静默死亡; ② config 跨仓引用 D:/Desktop/k9-training-system/.../smal_npy 实测不存在——按入口脚本容错设计不阻塞主曲线, 已在 Start-Q1 加三资源预检留痕(ckpt/pool_jsonl/smal_npy), 届时产物将缺真实池打分部分待用户裁决; ③ Start-Q1 引用作用域外 relayWait 变量——改读 state; 另补产物时间戳校验(LastWriteTime≥started_at 防旧报告误判完成)与 Q2 检测正则精确化。--help 冒烟/--fresh 参数/best.pt/pool_jsonl 实证在位; 修复后手动触发一轮全绿(state 保持 waiting 无误启动) |
| 08-24 20:42 | 异常: waiting 阶段发现 p05/c1 进程存活(非本窗口启动), 不干预只登记 | 100 %, 7653 MiB | PID=37156,14140 |
| 08-24 20:59 | 发现陌生训练进程(§6.1)——不杀不动, 上报待裁决, 本轮暂停接力 | 100 %, 7711 MiB | PID=11800:"C:\Program Files\Python312\python.exe" "D:\Desktop\知识库搭建\scripts\daemon.py" sta ; PID=4400:"C:\Program Files\Python312\python.exe" D:\Desktop\知识库搭建\scripts\mem0_mcp_server ; PID=4340:"C:\Program Files\Python312\python.exe" D:\Desktop\知识库搭建\scripts\mem0_mcp_server ; PID=12040:"C:\Program Files\Python312\python.exe" D:\Desktop\知识库搭建\scripts\mem0_mcp_server ; PID=17716:"C:\Program Files\Python312\python.exe" D:\Desktop\知识库搭建\scripts\mem0_mcp_server ; PID=30736:"C:\Program Files\Python312\python.exe" D:\Desktop\知识库搭建\scripts\mem0_mcp_server |
| 08-24 21:01 | 例行巡检: NTU 占用中(ntu_proc=2), 继续等待 | 100 %, 7683 MiB | 无条件变化, 不干预 |
| 08-24 21:12 | 例行巡检: NTU 占用中(ntu_proc=2), 继续等待 | 100 %, 7677 MiB | 无条件变化, 不干预 |
| 08-24 21:42 | 例行巡检: NTU 占用中(ntu_proc=2), 继续等待 | 96 %, 7680 MiB | 无条件变化, 不干预 |
| 08-24 22:12 | 例行巡检: NTU 占用中(ntu_proc=2), 继续等待 | 18 %, 7622 MiB | 无条件变化, 不干预 |
| 08-24 22:42 | 例行巡检: NTU 占用中(ntu_proc=2), 继续等待 | 98 %, 7629 MiB | 无条件变化, 不干预 |
| 08-24 23:10 | 例行巡检: NTU 占用中(ntu_proc=2), 继续等待 | 98 %, 7657 MiB | 无条件变化, 不干预 |
| 08-24 23:12 | 例行巡检: NTU 占用中(ntu_proc=2), 继续等待 | 100 %, 7644 MiB | 无条件变化, 不干预 |

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
| v1.1 | 2026-08-24 | 用户裁决 A：Q3a/Q3b/Q3c 并入队列（W20 C 路线三步接力），补 Q3 特别条款（内部串行+科学判读归 W20+失败禁改代码） |
