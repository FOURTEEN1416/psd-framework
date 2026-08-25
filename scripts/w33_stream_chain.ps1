# w33_stream_chain.ps1 — W33 三流补全自动链（用户裁决 A 后的执行件）
# 阶段: E2 bone_pretext(300ep) → E4 bone_LE(100ep) → E3 motion_pretext(300ep) → E5 motion_LE(100ep) → E6 3s融合
# 模式: 复刻 relay_executor 成熟模式——串行执行 + 内容级校验(非存在性) + 失败重试一次 + 再败 HALT 留证上板
# 状态: runs/w33_chain/state.json（幂等：已完成阶段跳过，可安全重启续跑）
# 用法: pwsh scripts/w33_stream_chain.ps1   （建议 Start-Process 分离运行）

param(
    [string]$Python = "D:\Desktop\psd-framework\.venv\Scripts\python.exe",
    [string]$RepoRoot = "D:\Desktop\psd-framework-W33"
)

$ErrorActionPreference = "Continue"
$stateDir = Join-Path $RepoRoot "runs\w33_chain"
$mainRepo = "D:\Desktop\psd-framework"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

function Write-ChainLog([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $msg
    Add-Content -Path (Join-Path $stateDir "chain.log") -Value $line
    Write-Host $line
}

function Update-State([hashtable]$st) {
    $st | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $stateDir "state.json") -Encoding utf8
}

function Publish-Board([string]$type, [string]$msg) {
    try {
        pwsh (Join-Path $mainRepo "scripts\window_board.ps1") -Append ("[{0}][_W33_] (ALL) {1}: {2}" -f (Get-Date -Format 'MM-dd HH:mm'), $type, $msg)
    } catch { Write-ChainLog "WARN: 看板写入失败: $_" }
}

# ---- 阶段定义 ----
$stages = @(
    @{ id="E2_bone_pretext";   entry="scripts/run_ntu_phaseb.py";      args="--config configs/ntu60_phaseb_pretext_xsub_bone.yaml";
      kind="train"; workdir="runs/ntu_phaseB/bone_pretext";   timeout_h=32 },
    @{ id="E4_bone_lineareval"; entry="scripts/run_ntu_lineareval.py";  args="--config configs/ntu60_phaseb_lineareval_xsub_bone.yaml";
      kind="lineareval"; workdir="runs/ntu_phaseB/lineareval_bone"; timeout_h=3 },
    @{ id="E3_motion_pretext"; entry="scripts/run_ntu_phaseb.py";      args="--config configs/ntu60_phaseb_pretext_xsub_motion.yaml";
      kind="train"; workdir="runs/ntu_phaseB/motion_pretext"; timeout_h=32 },
    @{ id="E5_motion_lineareval"; entry="scripts/run_ntu_lineareval.py"; args="--config configs/ntu60_phaseb_lineareval_xsub_motion.yaml";
      kind="lineareval"; workdir="runs/ntu_phaseB/lineareval_motion"; timeout_h=3 },
    @{ id="E6_ensemble_3s";    entry="scripts/ntu_ensemble_3s.py";     args="--json-out reports/ntu-phaseB-3s-ensemble.json";
      kind="ensemble"; workdir=$null; timeout_h=1 }
)

# ---- 状态装载（幂等续跑） ----
$statePath = Join-Path $stateDir "state.json"
if (Test-Path $statePath) {
    $st = Get-Content $statePath -Raw | ConvertFrom-Json
    Write-ChainLog "续跑：已存在状态文件（status=$($st.status)）"
} else {
    $st = @{
        chain = "W33-three-stream-A-decision"
        started_at = (Get-Date).ToString("o")
        status = "running"
        halt_stage = $null
        stages = @{}
    }
    Update-State $st
    Write-ChainLog "全新启动：用户裁决 A（补全三流），链式执行 $($stages.Count) 阶段"
}

# ---- 内容级校验器 ----
function Test-StageOutput($s) {
    switch ($s.kind) {
        "train" {
            $log = Join-Path $RepoRoot "$($s.workdir)\log.txt"
            if (-not (Test-Path $log)) { return @{ ok=$false; why="log.txt 不存在: $($s.workdir)" } }
            $tail = Get-Content $log -Tail 400
            $hitEpoch = ($tail | Select-String -SimpleMatch "Training epoch: 300").Count -gt 0
            if (-not $hitEpoch) {
                # 训练中或未跑满：区分"进行到多少轮"以便日志判读
                $lastEp = ($tail | Select-String -Pattern "Training epoch: (\d+)" | Select-Object -Last 1).Matches.Groups[1].Value
                return @{ ok=$false; why="未达 epoch300（当前可见最后轮次: $lastEp）" }
            }
            $ckpt = Join-Path $RepoRoot "$($s.workdir)\epoch300_model.pt"
            if (-not (Test-Path $ckpt)) { return @{ ok=$false; why="epoch300_model.pt 缺失" } }
            if ((Get-Item $ckpt).Length -lt 10MB) { return @{ ok=$false; why="checkpoint 小于 10MB，疑似损坏" } }
            $nanHit = ($tail | Select-String -Pattern "nan" -CaseSensitive:$false).Count -gt 0
            if ($nanHit) { return @{ ok=$false; why="日志尾部检出 nan（坍缩信号），拒绝放行" } }
            return @{ ok=$true; why="300/300 + ckpt 完整 + 无 nan" }
        }
        "lineareval" {
            $log = Join-Path $RepoRoot "$($s.workdir)\log.txt"
            if (-not (Test-Path $log)) { return @{ ok=$false; why="log.txt 不存在: $($s.workdir)" } }
            $m = Select-String -Path $log -Pattern "Best Top1: ([\d.]+)%" | Select-Object -Last 1
            if (-not $m) { return @{ ok=$false; why="未见 Best Top1 行（评估未完成？）" } }
            $pkl = Join-Path $RepoRoot "$($s.workdir)\test_result.pkl"
            if (-not (Test-Path $pkl)) { return @{ ok=$false; why="test_result.pkl 缺失（三流融合将不可用）" } }
            return @{ ok=$true; why="best_top1=$($m.Matches.Groups[1].Value)%; test_result.pkl 在案"; value=[double]$m.Matches.Groups[1].Value }
        }
        "ensemble" {
            $j = Join-Path $RepoRoot "reports\ntu-phaseB-3s-ensemble.json"
            if (-not (Test-Path $j)) { return @{ ok=$false; why="融合 JSON 未产出" } }
            try {
                $d = Get-Content $j -Raw | ConvertFrom-Json
                if ($null -eq $d.top1) { return @{ ok=$false; why="JSON 无 top1 字段" } }
                foreach ($sName in @("joint","bone","motion")) {
                    if ($null -eq $d.per_stream.$sName.best_top1) { return @{ ok=$false; why="$sName 流 best_top1 收集失败" } }
                }
                return @{ ok=$true; why="top1=$([math]::Round($d.top1*100,2))% top5=$([math]::Round($d.top5*100,2))%（三流收集完整）";
                          top1=[math]::Round($d.top1*100,2); top5=[math]::Round($d.top5*100,2); n=$d.n }
            } catch { return @{ ok=$false; why="JSON 解析失败: $_" } }
        }
    }
}

# ---- 主循环 ----
foreach ($s in $stages) {
    $sid = $s.id
    $rec = $st.stages.$sid
    if ($rec -and $rec.status -eq "done") { Write-ChainLog "跳过已完成阶段: $sid"; continue }

    # 幂等保护：校验历史产物（如上次运行中途死掉但产物已齐）
    $pre = Test-StageOutput $s
    if ($pre.ok) {
        Write-ChainLog "$sid 历史产物校验通过（$($pre.why)），标记 done 免重跑"
        $st.stages | Add-Member -NotePropertyName $sid -NotePropertyValue (@{ status="done"; note=$pre.why; value=$pre.value }) -Force
        Update-State $st
        continue
    }

    $attempt = if ($rec -and $rec.attempts) { [int]$rec.attempts } else { 0 }
    $success = $false
    $lastWhy = $pre.why

    while ($attempt -lt 2 -and -not $success) {
        $attempt++
        $t0 = Get-Date
        Write-ChainLog "▶ $sid 第 ${attempt} 次尝试启动: python $($s.entry) $($s.args)"
        $outLog = Join-Path $stateDir "$sid.a$attempt.out.log"
        $errLog = Join-Path $stateDir "$sid.a$attempt.err.log"
        $p = Start-Process -FilePath $Python `
                -ArgumentList "`"$((Resolve-Path (Join-Path $RepoRoot $s.entry)).Path)`" $($s.args)" `
                -WorkingDirectory $RepoRoot `
                -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
                -PassThru -WindowStyle Hidden
        Write-ChainLog "  PID=$($p.Id)，超时上限 $($s.timeout_h)h"

        $exited = $p.WaitForExit([int]($s.timeout_h * 3600 * 1000))
        if (-not $exited) {
            taskkill /PID $p.Id /T /F 2>&1 | Out-Null
            $lastWhy = "超时（>$($s.timeout_h)h）强杀"
            Write-ChainLog "  ✗ $lastWhy"
        } elseif ($p.ExitCode -ne 0) {
            $lastWhy = "exit=$($p.ExitCode)；stderr 尾: $(Get-Content $errLog -Tail 3 -ErrorAction SilentlyContinue | Out-String)"
            Write-ChainLog "  ✗ $lastWhy"
        } else {
            $chk = Test-StageOutput $s
            if ($chk.ok) {
                $success = $true
                $wall = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
                Write-ChainLog "  ✓ $sid 完成（${wall}min）：$($chk.why)"
                $st.stages | Add-Member -NotePropertyName $sid -NotePropertyValue (@{
                    status="done"; attempts=$attempt; wall_min=$wall; note=$chk.why; value=$chk.value
                }) -Force
                Update-State $st
                if ($s.kind -eq "lineareval") {
                    Publish-Board "INFO" "W33 自动链: $sid 完成 best_top1=$($chk.value)%（内容级校验通过，详见 runs/w33_chain/state.json）"
                }
                if ($s.kind -eq "ensemble") {
                    Publish-Board "INFO" ("W33 自动链 ALL_DONE: 3s 融合 top1={0}% top5={1}% (n={2}) —— R4 预注册线 ≥77.18% 判定素材就绪, 报告待人工综合归档" -f $chk.top1, $chk.top5, $chk.n)
                }
            } else {
                $lastWhy = $chk.why
                Write-ChainLog "  ✗ 内容级校验失败: $($chk.why)"
            }
        }
        if (-not $success -and $attempt -lt 2) { Write-ChainLog "  → 按协议重试一次" }
    }

    if (-not $success) {
        $st.status = "HALT"; $st.halt_stage = $sid
        $st.stages | Add-Member -NotePropertyName $sid -NotePropertyValue (@{
            status="failed"; attempts=$attempt; note=$lastWhy }) -Force
        Update-State $st
        Write-ChainLog "■ HALT @ $sid —— 重试已用尽，留证停链（state.json + 阶段日志在 runs/w33_chain/）"
        Publish-Board "BLOCK" "W33 自动链 HALT @ $sid：$lastWhy —— 证据 runs/w33_chain/（state.json/chain.log/阶段stdout+stderr），请协调者判读；后续阶段未执行"
        exit 2
    }
}

$st.status = "ALL_DONE"; $st.finished_at = (Get-Date).ToString("o")
Update-State $st
Write-ChainLog "■ 五阶段全部完成（ALL_DONE）——等待人工综合报告归档与收编"
