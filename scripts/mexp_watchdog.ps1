# MEXP watchdog (SYSTEM, every 10 min) - TRANS-002 MEXP pipeline guard
# Four branches: DONE/FATAL -> stop self; pipeline alive -> log; dead -> kill orphans + restart
$ErrorActionPreference = "Continue"
$repo = "D:\Desktop\psd-framework"
$stateFile = "$repo\reports\trans002_mexp_state.json"
$doneFile = "$repo\reports\trans002_mexp_DONE.flag"
$fatalFile = "$repo\reports\trans002_mexp_FATAL.txt"
$hbFile = "$repo\runs\watchdog_mexp_status.json"
$wdlog = "$repo\reports\trans002_mexp_watchdog.log"
$selfTask = "PSD_TRANS002_MEXP_Watchdog"
$mainTask = "PSD_TRANS002_MEXP"
$maxRestartsNoProgress = 5

function Log($m) {
    $line = "[$(Get-Date -Format s)] $m"
    $line | Out-File $wdlog -Append -Encoding utf8
}

function Write-HB($status, $detail) {
    @{ ts = (Get-Date -Format s); status = $status; detail = $detail } |
        ConvertTo-Json | Out-File $hbFile -Encoding utf8
}

if (Test-Path $doneFile) {
    Log "DONE present: MEXP finished; stopping watchdog task"
    Write-HB "done_stopped" "DONE flag present"
    schtasks /Change /TN $selfTask /DISABLE | Out-Null
    exit 0
}
if (Test-Path $fatalFile) {
    Log "FATAL present: unrecoverable; stopping watchdog task"
    Write-HB "fatal_stopped" (Get-Content $fatalFile -Raw)
    schtasks /Change /TN $selfTask /DISABLE | Out-Null
    exit 0
}

$alive = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match 'mexp_trans002\.py' }
if ($alive) {
    Log ("alive: MEXP pid(s) " + (($alive | ForEach-Object { $_.ProcessId }) -join ","))
    Write-HB "alive" ("pids=" + (($alive | ForEach-Object { $_.ProcessId }) -join ","))
    exit 0
}

$restartCount = 0
$progressHistory = @()
if (Test-Path "$repo\reports\trans002_mexp_restart_history.json") {
    $h = Get-Content "$repo\reports\trans002_mexp_restart_history.json" -Raw | ConvertFrom-Json
    $restartCount = $h.count
    $progressHistory = $h.attempts
}
$progressSig = ""
if (Test-Path $stateFile) {
    $st = Get-Content $stateFile -Raw | ConvertFrom-Json
    $v = @($st.pairs)
    $nf = @($st.nan_failed)
    $progressSig = ("pairs=" + $v.Count + ";nan=" + $nf.Count)
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

$orphans = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match 'run_r23_trans002' }
foreach ($o in $orphans) {
    Log ("killing orphan trainer pid=" + $o.ProcessId)
    Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue
}

$progressHistory += @{ sig = $progressSig; ts = (Get-Date -Format s) }
@{ count = $restartCount + 1; attempts = $progressHistory } |
    ConvertTo-Json -Depth 4 | Out-File "$repo\reports\trans002_mexp_restart_history.json" -Encoding utf8

Log ("MEXP pipeline dead; restarting (attempt " + ($restartCount + 1) + ", sig=$progressSig)")
schtasks /Run /TN $mainTask | Out-Null
Write-HB "restarted" ("attempt=" + ($restartCount + 1) + "; sig=$progressSig")
