# P1 冲刺实验队列发射器（单实例、串行、可监控、可中断）
# 用法: pwsh -File run_queue.ps1 [-Stop] [-Status]
# 任务定义见同目录 queue.json；每个任务状态落盘到 runs/p1_sprint/queue/<id>.status.json
param(
    [switch]$Stop,
    [switch]$Status
)
$ErrorActionPreference = 'Stop'
$queueSpecPath = Join-Path $PSScriptRoot 'queue.json'
$spec = Get-Content $queueSpecPath -Raw | ConvertFrom-Json
$queueDir = $spec.queueDir
$logDir = $spec.logDir
New-Item -ItemType Directory -Force -Path $queueDir | Out-Null
$lockPath = Join-Path $queueDir 'queue.lock'
$statusPath = Join-Path $queueDir 'runner.status.json'

function Write-Status([string]$state, [string]$jobId, [int]$pidArg) {
    $s = [ordered]@{
        runner_pid = $PID
        state      = $state
        job        = $jobId
        job_pid    = $pidArg
        updated_at = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    }
    $s | ConvertTo-Json | Set-Content -Path $statusPath -Encoding UTF8
}

if ($Status) {
    if (Test-Path $statusPath) { Get-Content $statusPath -Raw; exit 0 }
    'NO_RUNNER_STATUS_FILE'; exit 1
}

# 单实例锁：已有 runner 活着则不重复发射
if (Test-Path $lockPath) {
    $lockOwner = 0
    try { $lockOwner = [int](Get-Content $lockPath -Raw) } catch {}
    $alive = $false
    if ($lockOwner -gt 0) {
        try { $p = Get-Process -Id $lockOwner -ErrorAction Stop; if (-not $p.HasExited) { $alive = $true } } catch { $alive = $false }
    }
    if ($alive) {
        Write-Output "QUEUE_ALREADY_RUNNING pid=$lockOwner"
        exit 0
    }
    Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
}

# Stop：杀 runner 及其当前 job 子进程（taskkill /T 连树）
if ($Stop) {
    if (-not (Test-Path $statusPath)) { Write-Output 'NO_RUNNER_TO_STOP'; exit 1 }
    $st = Get-Content $statusPath -Raw | ConvertFrom-Json
    $runnerPid = [int]$st.runner_pid
    try {
        taskkill /PID $runnerPid /T /F 2>$null | Out-Null
        Write-Output "STOP_SIGNAL_SENT runner=$runnerPid"
    } catch {
        Write-Output "STOP_FAILED runner=$runnerPid : $($_.Exception.Message)"
        exit 1
    }
    exit 0
}

# ---- 正常发射路径 ----
[System.IO.File]::WriteAllText($lockPath, "$PID")
$python = $spec.python
$cwd = $spec.cwd

# 让 PYTHONPATH/临时环境变量优先，避免污染本仓 import
$env:PYTHONNOUSERSITE = '1'

foreach ($job in $spec.jobs) {
    $id = $job.id
    $log = Join-Path $logDir ($id + '.log')
    $stFile = Join-Path $queueDir ($id + '.status.json')

    # 幂等：已有 summary.json（全量产物）则跳过，除非强制
    if (Test-Path $job.summary) {
        $st = [ordered]@{ id = $id; state = 'skipped_already_done'; log = $log; started_at = $null; ended_at = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'); exit_code = $null }
        $st | ConvertTo-Json | Set-Content $stFile -Encoding UTF8
        Write-Output "SKIP $id (summary exists)"
        continue
    }

    Write-Status 'running' $id 0
    $startInfo = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $id`nCMD: `"$python`" `"$($job.script)`" $($job.args -join ' ')`n"
    [System.IO.File]::WriteAllText($log, $startInfo)

    $p = Start-Process -FilePath $python -ArgumentList (@($job.script) + @($job.args)) -WorkingDirectory $cwd -RedirectStandardOutput $log -RedirectStandardError ($log + '.err') -NoNewWindow -PassThru
    Write-Status 'running' $id $p.Id
    try { $p.WaitForExit() } catch {}
    $code = $p.ExitCode
    $tail = ''
    try {
        $content = (Get-Content $log -Raw -ErrorAction Stop)
        if ($content.Contains($job.doneMarker)) { $code = 0; $note = 'marker_ok' } else { $note = '' }
        $lines = $content -split "`r?`n"
        $tail = ($lines | Select-Object -Last 6) -join "`n"
    } catch { $note = '' }
    $state = if ($code -eq 0) { 'done' } else { 'failed' }
    $st = [ordered]@{ id = $id; state = $state; exit_code = $code; log = $log; started_at = $null; ended_at = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') }
    $st | ConvertTo-Json | Set-Content $stFile -Encoding UTF8
    Add-Content -Path $log -Value "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $id exit=$code $note"
    Write-Output "FINISHED $id exit=$code"
    Write-Status 'between_jobs' '' 0
    Start-Sleep -Seconds ([int]$spec.idleSecondsBetweenJobs)
}
Write-Status 'all_done' '' 0
Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
Write-Output 'QUEUE_ALL_DONE'
