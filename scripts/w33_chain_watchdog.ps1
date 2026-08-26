# w33_chain_watchdog.ps1 — 三流链看门狗（Task Scheduler 幸存者模式）
# 逻辑: 融合 JSON 未产出 且 无 ntu_phaseb 训练进程 → 调用链脚本(内部状态机会从失败阶段重试)
#       单实例防护: 有活进程绝不重复点火(01:42 双启动事故的根治)
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

# 单实例防护: 有 ntu_phaseb 训练进程在跑则不动
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "ntu_phaseb" }
if ($procs) {
    Log ("训练进程存活({0})，不干预" -f ($procs.ProcessId -join ','))
    exit 0
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
