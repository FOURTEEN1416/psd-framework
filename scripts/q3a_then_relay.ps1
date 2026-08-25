# q3a_then_relay.ps1 — Q3a 独立进程跑法(绕开隐藏窗原生崩溃) + 成功后链入 Q3b/Q3c
# Q3a 日志独立落盘 runs/q3a_console.log(不依赖 transcript,崩溃可取证)

Set-Location "D:\Desktop\psd-framework"
$PY = "D:\Desktop\psd-framework\.venv\Scripts\python.exe"
$log = "D:\Desktop\psd-framework\runs\q3a_console.log"

Write-Host "[q3a] 独立启动 YOLO 训练,日志 -> $log"
$p = Start-Process -FilePath $PY `
    -ArgumentList "scripts\train_yolo_dogpose.py","--epochs","50","--batch","16" `
    -WorkingDirectory "D:\Desktop\psd-framework" `
    -RedirectStandardOutput $log -RedirectStandardError "D:\Desktop\psd-framework\runs\q3a_console.err.log" `
    -PassThru -WindowStyle Hidden
Write-Host "[q3a] PID=$($p.Id)"

$p.WaitForExit()
$code = $p.ExitCode
Write-Host "[q3a] 退出码=$code"

$best = "D:\Desktop\psd-framework\runs\public_real_yolo_dogpose\train\weights\best.pt"
if ($code -eq 0 -and (Test-Path $best) -and ((Get-Item $best).Length -gt 10MB)) {
    Write-Host "[q3a] 成功,提交证据"
    Set-Location "D:\Desktop\psd-framework"
    git add runs/public_real_yolo_dogpose 2>&1 | Out-Null
    git commit -m "feat(p05): [relay Q3a] YOLO dog-pose 24 点权重(独立进程跑法)" 2>&1 | Out-Null
    Write-Host "[q3a] 链入 relay -StartStep Q3b_extract"
    & pwsh -NoProfile -ExecutionPolicy Bypass -File "D:\Desktop\psd-framework\scripts\relay_executor.ps1" -StartStep "Q3b_extract"
} else {
    Write-Host "[q3a] 失败(码=$code)——日志尾部:"
    Get-Content $log -Tail 30 -ErrorAction SilentlyContinue
    @{ halted_at = (Get-Date -Format o); step = "Q3a_standalone"; exit = $code } |
        ConvertTo-Json | Out-File "D:\Desktop\psd-framework\runs\relay_exec\HALT.json" -Encoding utf8
    Write-Host "[q3a] HALT 留证,待人工"
}
