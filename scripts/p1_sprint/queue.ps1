# P1 队列监控/停止工具
# 用法:
#   pwsh -File queue.ps1                 # 看总览（runner + 各任务状态 + 日志尾部）
#   pwsh -File queue.ps1 -Stop           # 停止 runner 及当前任务（连进程树）
#   pwsh -File queue.ps1 -Tail <job_id>  # 实时跟某个任务日志（Ctrl+C 退出）
param(
    [switch]$Stop,
    [string]$Tail
)
$ErrorActionPreference = 'Stop'
$spec = Get-Content (Join-Path $PSScriptRoot 'queue.json') -Raw | ConvertFrom-Json
$queueDir = $spec.queueDir
$logDir = $spec.logDir
$statusPath = Join-Path $queueDir 'runner.status.json'

if ($Stop) {
    & (Join-Path $PSScriptRoot 'run_queue.ps1') -Stop
    exit $LASTEXITCODE
}

if ($Tail) {
    $log = Join-Path $logDir ($Tail + '.log')
    if (-not (Test-Path $log)) { Write-Output "NO_LOG $log"; exit 1 }
    Get-Content $log -Tail 40 -Wait
    exit 0
}

# ---- 总览 ----
Write-Output "=== RUNNER ==="
if (Test-Path $statusPath) {
    $st = Get-Content $statusPath -Raw | ConvertFrom-Json
    $alive = $false
    try { $p = Get-Process -Id $st.runner_pid -ErrorAction Stop; $alive = -not $p.HasExited } catch {}
    Write-Output ("runner_pid={0} alive={1} state={2} job={3} job_pid={4} updated={5}" -f $st.runner_pid, $alive, $st.state, $st.job, $st.job_pid, $st.updated_at)
} else {
    Write-Output 'runner 未发射（无状态文件）'
}
Write-Output "`n=== JOBS ==="
foreach ($job in $spec.jobs) {
    $id = $job.id
    $stFile = Join-Path $queueDir ($id + '.status.json')
    $log = Join-Path $logDir ($id + '.log')
    $summaryDone = Test-Path $job.summary
    $line = $id.PadRight(28)
    if ($summaryDone) { $line += 'summary.json 已存在(完成) ' }
    if (Test-Path $stFile) {
        $js = Get-Content $stFile -Raw | ConvertFrom-Json
        $line += ("state={0} exit={1}" -f $js.state, $js.exit_code)
    } else {
        $line += 'state=pending'
    }
    Write-Output $line
    if (Test-Path $log) {
        $last = Get-Content $log -Tail 3 -ErrorAction SilentlyContinue
        foreach ($l in $last) { Write-Output ("    | " + $l) }
    }
    Write-Output ""
}
