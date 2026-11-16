# P1 E3b k 阶梯多种子补测发射器（seeds 43,44 × k=10/25/50 × arms decoupled,frozen_head）
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/p1_sprint/run_k_ladder_seeds.ps1
# 日志: runs/p1_sprint/k_ladder_seeds/k_ladder_s43_44_k102550.log；summary: runs/p1_sprint/k_ladder/summary.json
# 说明: 发射前等待 GPU 空闲（memory.used<=3000MiB，避免与并行任务争抢 8GB 显存）；
#       幂等 = 日志已含 K_LADDER_DONE 则跳过；完成后 doneMarker 验收写状态文件。
$ErrorActionPreference = 'Stop'
$repo = 'D:\Desktop\PR论文\psd-framework'
$py = Join-Path $repo '.venv\Scripts\python.exe'
$queueDir = Join-Path $repo 'runs\p1_sprint\queue'
$seedDir = Join-Path $repo 'runs\p1_sprint\k_ladder_seeds'
New-Item -ItemType Directory -Force -Path $queueDir, $seedDir | Out-Null
$lockPath = Join-Path $queueDir 'k_ladder_seeds.lock'
$statusPath = Join-Path $queueDir 'k_ladder_seeds.status.json'

$id = 'k_ladder_s43_44_k102550'
$log = Join-Path $seedDir ($id + '.log')
$marker = 'K_LADDER_DONE'

# 单实例锁
if (Test-Path $lockPath) {
    $owner = 0
    try { $owner = [int](Get-Content $lockPath -Raw) } catch {}
    $alive = $false
    if ($owner -gt 0) {
        try { $p = Get-Process -Id $owner -ErrorAction Stop; if (-not $p.HasExited) { $alive = $true } } catch {}
    }
    if ($alive) { Write-Output "K_LADDER_SEEDS_ALREADY_RUNNING pid=$owner"; exit 0 }
    Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
}
[System.IO.File]::WriteAllText($lockPath, "$PID")

# 幂等：本任务日志已含 doneMarker 则跳过
if ((Test-Path $log) -and (Get-Content $log -Raw -ErrorAction SilentlyContinue).Contains($marker)) {
    [ordered]@{ id=$id; state='skipped_already_done'; ended_at=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss') } |
        ConvertTo-Json | Set-Content $statusPath -Encoding UTF8
    Write-Output "SKIP $id (marker in log)"
    Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
    exit 0
}

# 等待 GPU 空闲（8GB 卡，避免挤压并行任务；30s 轮询，上限 90 分钟）
$gpuWaitSec = 0
while ($true) {
    $used = 99999
    try {
        $line = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1
        $used = [int]($line -replace '[^0-9]', '')
    } catch {}
    if ($used -le 3000) { break }
    if ($gpuWaitSec -ge 5400) {
        [ordered]@{ id=$id; state='gpu_busy_timeout'; gpu_used_mib=$used; waited_sec=$gpuWaitSec;
                    ended_at=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss') } |
            ConvertTo-Json | Set-Content $statusPath -Encoding UTF8
        Write-Output "GPU_BUSY_TIMEOUT used=${used}MiB"
        Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
        exit 2
    }
    if ($gpuWaitSec -eq 0 -or ($gpuWaitSec % 300) -eq 0) {
        Write-Output "WAIT_GPU used=${used}MiB waited=${gpuWaitSec}s"
    }
    Start-Sleep -Seconds 30
    $gpuWaitSec += 30
}
Write-Output "GPU_FREE used=${used}MiB (waited ${gpuWaitSec}s)"

$env:PYTHONNOUSERSITE = '1'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

[ordered]@{ runner_pid=$PID; state='running'; job=$id; job_pid=0;
            gpu_waited_sec=$gpuWaitSec; updated_at=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss') } |
    ConvertTo-Json | Set-Content $statusPath -Encoding UTF8

$args = @('-u', 'scripts/p1_sprint/k_ladder.py',
          '--seeds', '43,44',
          '--k-values', '10,25,50',
          '--arms', 'decoupled,frozen_head')
$p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $repo `
        -RedirectStandardOutput $log -RedirectStandardError ($log + '.err') -NoNewWindow -PassThru
[ordered]@{ runner_pid=$PID; state='running'; job=$id; job_pid=$p.Id;
            updated_at=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss') } |
    ConvertTo-Json | Set-Content $statusPath -Encoding UTF8
Write-Output "LAUNCHED $id pid=$($p.Id)"

try { $p.WaitForExit() } catch {}
$code = $p.ExitCode
$note = ''
try {
    $content = Get-Content $log -Raw -ErrorAction Stop
    if ($content.Contains($marker)) { $code = 0; $note = 'marker_ok' }
} catch {}
$state = if ($code -eq 0) { 'done' } else { 'failed' }
[ordered]@{ id=$id; state=$state; exit_code=$code; log=$log; ended_at=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss') } |
    ConvertTo-Json | Set-Content $statusPath -Encoding UTF8
Add-Content -Path $log -Value "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $id exit=$code $note"
Write-Output "FINISHED $id exit=$code $note"

Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
Write-Output 'K_LADDER_SEEDS_RUNNER_DONE'
