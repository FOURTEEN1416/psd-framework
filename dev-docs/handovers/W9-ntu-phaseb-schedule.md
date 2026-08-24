# NTU Phase B 夜间排程任务书

> 窗口: 定时任务 / W9 授权代表
> 依据: HANDOVER v1.5 §8 — "Phase B 复现训练 GPU 已解禁可排程（P0.2 冲刺收官释放显卡；错峰默认=夜间长训练，与 W11 白天冒烟互斥让行）"
> 当前状态: W12 占用 GPU 白天时段（2026-08-24 11:xx），夜间 23:00 后空闲

## 启动条件

满足以下全部条件才启动：
1. `nvidia-smi` 显示 GPU 利用率 < 10% 且显存占用 < 500 MiB
2. 无 `train_smq_segmentation.py` / `run_p05_full.py` 等 psd-framework 训练进程
3. 当前时间 ≥ 23:00（避免与 W12 白天工作冲突）

## 执行流程

```powershell
# 1. 启动 NTU Phase B 复现训练（后台分离）
$env:CUDA_VISIBLE_DEVICES=''  # 或使用默认 GPU
Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "scripts/run_ntu_phaseb.py" `
  -WorkingDirectory "D:\Desktop\psd-framework" `
  -RedirectStandardOutput "runs/ntu_phaseB/console_out.log" `
  -RedirectStandardError "runs/ntu_phaseB/console_err.log" `
  -PassThru -WindowStyle Hidden

# 2. 日志监控（每 30 分钟一次，最多 8 次 = 4 小时）
for ($i = 0; $i -lt 8; $i++) {
  Start-Sleep -Seconds 1800
  $log = Get-Content runs/ntu_phaseB/console_err.log -Tail 3 -ErrorAction SilentlyContinue
  Write-Host "[$(Get-Date -Format 'HH:mm:ss')] NTU Phase B: $log"
  if (-not (Get-Process -Id <PID> -ErrorAction SilentlyContinue)) {
    Write-Host "NTU Phase B 进程已退出"
    break
  }
}

# 3. 完成后写摘要
$summary = @{
  status = "completed"
  start_time = "2026-08-24T23:00:00"
  end_time = (Get-Date).ToString("o")
  final_metrics = @{}  # 从 console_out.log 提取
}
$summary | ConvertTo-Json | Out-File reports/ntu-phaseB-schedule-summary.json -Encoding UTF8
```

## 失败熔断

- 连续 3 次 epoch loss NaN → 立即终止，记录 `reports/ntu-phaseB-crash-evidence.md`，上报用户裁决
- GPU OOM → 检查 batch_size 配置，降档重试（最多 2 次）
- 进程意外退出 → 检查 console_err.log 最后 50 行，判断是代码 bug 还是资源问题

## 完成后复核

1. 确认 checkpoint 落盘 `runs/ntu_phaseB/models/best.pt`
2. 确认 metrics JSON 写入 `reports/ntu-phaseB-results.json`
3. 更新 `dev-docs/HANDOVER.md` W9 行状态
4. Commit: `feat(wip): W9 NTU Phase B 夜间训练完成——<数字摘要>`

## 注意事项

- 本任务只读消费 NTU 数据（不修改数据文件）
- 不安装新依赖（复用 .venv 已有包）
- 与 W12 的 P0.5 实验无文件冲突（不同 runs/ 子目录）
- 若 W12 夜间也需 GPU，本任务自动跳过（启动条件第 1 条不满足）
