# w33_chain_watchdog.ps1 — 三流链看门狗（Task Scheduler 幸存者模式）
# 逻辑: 融合 JSON 未产出 且 无 ntu_phaseb 训练进程 → 调用链脚本(内部状态机会从失败阶段重试)
#       单实例防护: 有活进程绝不重复点火(01:42 双启动事故的根治)
#       活性校验(v2, 2026-08-27): 进程存活但训练日志静默>=30min 不算健康——连续两轮巡检仍静默才 kill+重启。
#           教训固化的两条事故依据: ①08-27晨 CPU 快照在权限不足时读到假 0 导致误判僵尸;
#           ②RAM 枯竭时段会出现小时级深度停顿(单轮 57min 实测), 单次静默不足以判死。
# 完成检测: W33 worktree 出现 3s 融合 JSON → 自注销

$ErrorActionPreference = "Continue"
$W33 = "D:\Desktop\psd-framework-W33"
$fusion = Join-Path $W33 "reports\ntu-phaseB-3s-ensemble.json"
$chainLog = Join-Path $W33 "runs\w33_chain\watchdog.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $msg
    Add-Content -LiteralPath $chainLog -Value $line -Encoding UTF8
}

# 完成检测 → 自注销
if (Test-Path $fusion) {
    Log "融合 JSON 已产出——三流链完成，看门狗自注销"
    Unregister-ScheduledTask -TaskName "w33-chain-watchdog" -TaskPath "\OpenCode\" -Confirm:$false -ErrorAction SilentlyContinue
    exit 0
}

# 单实例防护: 有 ntu_phaseb 训练进程在跑则进入活性校验(v2), 否则继续 HALT 检查
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "ntu_phaseb" }
if ($procs) {
    # --- v2 活性校验: 全局最新训练日志 mtime ---
    $logs = Get-ChildItem (Join-Path $W33 "runs\ntu_phaseB") -Recurse -Filter "log.txt" -ErrorAction SilentlyContinue
    $newest = $logs | Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $newest) {
        Log ("训练进程存活({0}) 但未找到任何 log.txt——保守放行不干预" -f ($procs.ProcessId -join ','))
        exit 0
    }
    $silentMin = ((Get-Date) - $newest.LastWriteTime).TotalMinutes
    if ($silentMin -lt 30) {
        Log ("训练进程存活({0}) 日志新鲜({1:n0}min 前: {2}) 不干预" -f ($procs.ProcessId -join ','), $silentMin, $newest.Directory.Name)
        exit 0
    }
    # 静默 >=30min: 双确认状态机(silence_flag 由上一轮巡检落盘)
    $flag = Join-Path $W33 "runs\w33_chain\silence_flag"
    if (Test-Path $flag) {
        Log ("ALERT: 连续两轮静默确认({0:n0}min 无任何 iter 输出, 最新={1})——判定 stuck, kill 后点火重启" -f $silentMin, $newest.FullName)
        $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 5
        Remove-Item $flag -Force -ErrorAction SilentlyContinue
        # 继续向下走到点火段(HALT 检查仍生效)
    } else {
        New-Item $flag -ItemType File -Force | Out-Null
        Log ("警告: 训练静默 {0:n0}min(最新输出={1}), 首次标记——下一轮巡检({2})仍静默则自动 kill+重启" -f $silentMin, $newest.FullName, (Get-Date).AddMinutes(20).ToString('HH:mm'))
        exit 0
    }
} else {
    # 无进程时清理可能残留的静默标记, 避免下次误触发双确认
    Remove-Item (Join-Path $W33 "runs\w33_chain\silence_flag") -Force -ErrorAction SilentlyContinue
}

# HALT 检查: 链自身判死则不自动重试(留人工)
$stateFile = Join-Path $W33 "runs\w33_chain\state.json"
if (Test-Path $stateFile) {
    try {
        $st = Get-Content $stateFile -Raw | ConvertFrom-Json
        if ($st.halt_stage) {
            Log ("链处于 HALT({0})——不自动重试，待人工" -f $st.halt_stage)
            exit 0
        }
    } catch { Log "state.json 解析失败，按可重试处理" }
}

# 点火: 调用链脚本(其内部状态机会从死亡阶段重试; 骨干无断点续训则该阶段从头)
Log "无训练进程且链未完成——重新点火链脚本"
Start-Process -FilePath "pwsh" `
    -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",(Join-Path $W33 "scripts\w33_stream_chain.ps1") `
    -WorkingDirectory $W33 -WindowStyle Hidden
Log "链脚本已点火"
