# 后台发射 P1 队列（detached，随窗口会话存活；不依赖任务窗口退出）
# 用法: pwsh -File launch_queue.ps1
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run_queue.ps1'
$out = Join-Path $PSScriptRoot '..\..\runs\p1_sprint\queue\launcher.out.log'
$err = Join-Path $PSScriptRoot '..\..\runs\p1_sprint\queue\launcher.err.log'
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
$p = Start-Process -FilePath 'pwsh' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runner) -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
Write-Output "LAUNCHED runner pid=$($p.Id)"
