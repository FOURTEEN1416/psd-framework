# NTU Phase B 排程任务书（白天+夜间均可）

> 窗口: 定时任务 / W9 授权代表
> 依据: HANDOVER v1.5 §8 + 用户 2026-08-24 裁决"白天也能跑"
> 当前状态: W12/A 占 GPU ~30%/2.5GB，RTX 5060 8GB 有充足余量并行

## 启动条件

满足以下全部条件才启动：
1. `nvidia-smi` 显存占用 < 6 GB（预留 ≥2 GB 给 NTU 模型）
2. 无 NTU 同名进程已在运行（防重复启动）
3. ~~当前时间 ≥ 23:00~~ **已取消时间限制**——白天夜间均可

## 执行流程

```powershell
# 1. 启动 NTU Phase B 复现训练（后台分离）
Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "scripts/run_ntu_phaseb.py" `
  -WorkingDirectory "D:\Desktop\psd-framework" `
  -RedirectStandardOutput "runs/ntu_phaseB/console_out.log" `
  -RedirectStandardError "runs/ntu_phaseB/console_err.log" `
  -PassThru -WindowStyle Hidden

# 2. 日志监控（每 30 分钟一次，最多 16 次 = 8 小时）
for ($i = 0; $i -lt 16; $i++) {
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
  start_time = (Get-Date).ToString("o")
  end_time = (Get-Date).ToString("o")
  final_metrics = @{}
}
$summary | ConvertTo-Json | Out-File reports/ntu-phaseB-schedule-summary.json -Encoding UTF8
```

## 失败熔断

- 连续 3 次 epoch loss NaN → 立即终止，记录 `reports/ntu-phaseB-crash-evidence.md`，上报用户裁决
- GPU OOM → 等待其他进程释放后自动重试（最多 3 次，间隔 10 分钟）
- 进程意外退出 → 检查 console_err.log 最后 50 行，判断是代码 bug 还是资源问题

## 完成后复核

1. 确认 checkpoint 落盘 `runs/ntu_phaseB/models/best.pt`
2. 确认 metrics JSON 写入 `reports/ntu-phaseB-results.json`
3. 更新 `dev-docs/HANDOVER.md` W9 行状态
4. Commit: `feat(wip): W9 NTU Phase B 训练完成——<数字摘要>`

## 注意事项

- 本任务只读消费 NTU 数据（不修改数据文件）
- 不安装新依赖（复用 .venv 已有包）
- 与 W12 的 P0.5 实验无文件冲突（不同 runs/ 子目录）
- 与 W12 共享 GPU 时各自性能会降低 ~50%，但均可正常完成
