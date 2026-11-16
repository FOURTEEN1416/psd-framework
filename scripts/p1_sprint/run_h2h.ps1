# P1 头对头基线串行发射器（published_baseline -> supervised_stgcn，单实例锁，doneMarker 验收）
# 用法: pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/p1_sprint/run_h2h.ps1
# 日志: runs/p1_sprint/<id>.log (+ .err)；状态: runs/p1_sprint/queue/h2h_<id>.status.json
$ErrorActionPreference = 'Stop'
$repo = 'D:\Desktop\PR论文\psd-framework'
$py = Join-Path $repo '.venv\Scripts\python.exe'
$logDir = Join-Path $repo 'runs\p1_sprint'
$queueDir = Join-Path $logDir 'queue'
New-Item -ItemType Directory -Force -Path $queueDir | Out-Null
$lockPath = Join-Path $queueDir 'h2h.lock'
$statusPath = Join-Path $queueDir 'h2h_runner.status.json'

# 单实例锁：已有 runner 活着则不重复发射
if (Test-Path $lockPath) {
    $owner = 0
    try { $owner = [int](Get-Content $lockPath -Raw) } catch {}
    $alive = $false
    if ($owner -gt 0) {
        try { $p = Get-Process -Id $owner -ErrorAction Stop; if (-not $p.HasExited) { $alive = $true } } catch {}
    }
    if ($alive) { Write-Output "H2H_ALREADY_RUNNING pid=$owner"; exit 0 }
    Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
}
[System.IO.File]::WriteAllText($lockPath, "$PID")
$env:PYTHONNOUSERSITE = '1'
# 8GB 笔记本 GPU 显存碎片缓解（ST-GCN 监督训练 einsum 中间量）
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

$jobs = @(
    @{ id='published_baseline_ntu60'; script='scripts/p1_sprint/published_baseline_ntu60.py';
       args=@();
       marker='P1_SPRINT_PUBLISHED_AIMCLR_NTU60_DONE'; summary='runs/p1_sprint/published_baseline_ntu60/summary.json' },
    @{ id='supervised_stgcn_ntu60'; script='scripts/p1_sprint/supervised_stgcn_ntu60.py';
       args=@('--hidden-channels','16');
       marker='P1_SPRINT_SUPERVISED_STGCN_NTU60_DONE'; summary='runs/p1_sprint/supervised_stgcn_ntu60/summary.json' }
)

foreach ($job in $jobs) {
    $log = Join-Path $logDir ($job.id + '.log')
    $stFile = Join-Path $queueDir ('h2h_' + $job.id + '.status.json')
    $summaryPath = Join-Path $repo $job.summary

    # 幂等：全量 summary 已存在则跳过
    if (Test-Path $summaryPath) {
        [ordered]@{ id=$job.id; state='skipped_already_done'; ended_at=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss') } |
            ConvertTo-Json | Set-Content $stFile -Encoding UTF8
        Write-Output "SKIP $($job.id) (summary exists)"
        continue
    }

    [ordered]@{ runner_pid=$PID; state='running'; job=$job.id; job_pid=0; updated_at=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss') } |
        ConvertTo-Json | Set-Content $statusPath -Encoding UTF8

    $p = Start-Process -FilePath $py -ArgumentList (@('-u', $job.script) + @($job.args)) -WorkingDirectory $repo `
            -RedirectStandardOutput $log -RedirectStandardError ($log + '.err') -NoNewWindow -PassThru
    [ordered]@{ runner_pid=$PID; state='running'; job=$job.id; job_pid=$p.Id; updated_at=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss') } |
        ConvertTo-Json | Set-Content $statusPath -Encoding UTF8
    Write-Output "LAUNCHED $($job.id) pid=$($p.Id)"

    try { $p.WaitForExit() } catch {}
    $code = $p.ExitCode
    $note = ''
    try {
        $content = Get-Content $log -Raw -ErrorAction Stop
        if ($content.Contains($job.marker)) { $code = 0; $note = 'marker_ok' }
    } catch {}
    $state = if ($code -eq 0) { 'done' } else { 'failed' }
    [ordered]@{ id=$job.id; state=$state; exit_code=$code; log=$log; ended_at=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss') } |
        ConvertTo-Json | Set-Content $stFile -Encoding UTF8
    Add-Content -Path $log -Value "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $($job.id) exit=$code $note"
    Write-Output "FINISHED $($job.id) exit=$code $note"
}

[ordered]@{ runner_pid=$PID; state='all_done'; updated_at=(Get-Date).ToString('yyyy-MM-dd HH:mm:ss') } |
    ConvertTo-Json | Set-Content $statusPath -Encoding UTF8
Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
Write-Output 'H2H_ALL_DONE'
