# TRANS-002 fixfull 看门狗（SYSTEM 每 10 分钟, 2026-09-08 晚上线）
# 四分支: DONE/FATAL 真终态→停自身任务; FATAL 存在→不重启(真终态, 留报告);
#         管线进程活着→仅记录; 进程死了且无终态标记→杀残留训练进程+重启管线(断点续训)。
# 限流: 连续重启 >=5 次且 state.json 无进展 → 停止重启并写 FATAL(防无限循环烧机)。
$ErrorActionPreference = "Continue"
$repo = "D:\Desktop\psd-framework"
$stateFile = "$repo\reports\trans002_fixfull_state.json"
$doneFile = "$repo\reports\trans002_fixfull_DONE.flag"
$fatalFile = "$repo\reports\trans002_fixfull_FATAL.txt"
$hbFile = "$repo\runs\watchdog_trans002_status.json"
$selfTask = "PSD_TRANS002_FixFull_Watchdog"
$mainTask = "PSD_TRANS002_FixFull"
$maxRestartsNoProgress = 5

function Log($m) {
    $line = "[$(Get-Date -Format s)] $m"
    $line | Out-File "$repo\reports\trans002_watchdog.log" -Append -Encoding utf8
}

function Write-HB($status, $detail) {
    @{ ts = (Get-Date -Format s); status = $status; detail = $detail } |
        ConvertTo-Json | Out-File $hbFile -Encoding utf8
}

# ---- 分支1: DONE 真终态 → 自杀停机 ----
if (Test-Path $doneFile) {
    Log "DONE present: pipeline finished; stopping watchdog task"
    Write-HB "done_stopped" "DONE flag present"
    schtasks /Change /TN $selfTask /DISABLE | Out-Null
    exit 0
}
# ---- 分支2: FATAL 真终态 → 停机(不重启, 等人工) ----
if (Test-Path $fatalFile) {
    Log "FATAL present: unrecoverable; stopping watchdog task"
    Write-HB "fatal_stopped" (Get-Content $fatalFile -Raw)
    schtasks /Change /TN $selfTask /DISABLE | Out-Null
    exit 0
}

# ---- 分支3: 管线进程存活检测（按命令行匹配, 非 PID 缓存）----
$alive = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match 'fixfull_trans002\.py' }
if ($alive) {
    Log ("alive: pipeline pid(s) " + (($alive | ForEach-Object { $_.ProcessId }) -join ","))
    Write-HB "alive" ("pids=" + (($alive | ForEach-Object { $_.ProcessId }) -join ","))
    exit 0
}

# ---- 分支4: 进程死亡且无终态标记 → 杀残留训练子进程 + 重启管线 ----
# 重启限流: state.json 的 attempt 在最近 maxRestartsNoProgress 次重启内无进展则放弃
$restartCount = 0
$progressHistory = @()
if (Test-Path "$repo\reports\trans002_restart_history.json") {
    $h = Get-Content "$repo\reports\trans002_restart_history.json" -Raw | ConvertFrom-Json
    $restartCount = $h.count
    $progressHistory = $h.attempts
}
$progressSig = ""
if (Test-Path $stateFile) {
    $st = Get-Content $stateFile -Raw | ConvertFrom-Json
    $v = @($st.progress.valid_seeds)
    $nf = @($st.progress.nan_failed)
    $progressSig = ("valid=" + $v.Count + ";nan=" + $nf.Count)
}
if ($progressHistory.Count -ge $maxRestartsNoProgress -and
    ($progressHistory | Select-Object -Last $maxRestartsNoProgress | ForEach-Object { $_.sig } | Select-Object -Unique).Count -eq 1 -and
    $progressSig -ne "" -and
    ($progressHistory | Select-Object -Last 1).sig -eq $progressSig) {
    Log "restart limit reached without progress; writing FATAL and stopping"
    "watchdog: $maxRestartsNoProgress restarts with no state progress (sig=$progressSig)" |
        Out-File $fatalFile -Encoding utf8
    schtasks /Change /TN $selfTask /DISABLE | Out-Null
    exit 1
}

# 杀可能残留的训练子进程（train_one / 其 GPU 子进程）——防止双训练抢 GPU
$orphans = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match 'run_r23_trans002' }
foreach ($o in $orphans) {
    Log ("killing orphan trainer pid=" + $o.ProcessId)
    Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue
}

# 记录重启历史（sig + 时间）
$progressHistory += @{ sig = $progressSig; ts = (Get-Date -Format s) }
@{ count = $restartCount + 1; attempts = $progressHistory } |
    ConvertTo-Json -Depth 4 | Out-File "$repo\reports\trans002_restart_history.json" -Encoding utf8

Log ("pipeline dead; restarting (attempt " + ($restartCount + 1) + ", sig=$progressSig)")
schtasks /Run /TN $mainTask | Out-Null
Write-HB "restarted" ("attempt=" + ($restartCount + 1) + "; sig=$progressSig")
