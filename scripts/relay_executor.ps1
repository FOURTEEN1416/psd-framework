# relay_executor.ps1 — PSD-Framework GPU 接力总执行器（协调者部署版）
# 队列: Q1(AL full) → Q2(C1 full) → Q3a(YOLO权重) → Q3b(全量提点) → Q3c(公开真实层微调)
# 门禁: NTU 释放后显存 <500MiB 且无 psd 训练进程持续 600s 才启动；步骤严格串行；
#       每步失败自动重试一次（间隔 120s），再失败即 HALT 留证。
# 日志: runs/relay_exec/transcript.log + 每步状态 runs/relay_exec/state.json

$ErrorActionPreference = "Continue"
Set-Location "D:\Desktop\psd-framework"
Start-Transcript -Path "runs\relay_exec\transcript.log" -Append | Out-Null

$PY = "D:\Desktop\psd-framework\.venv\Scripts\python.exe"
$DATE = "2026-08-25"
$state = @{ started_at = (Get-Date -Format o); steps = @() }
function Save-State { $state | ConvertTo-Json -Depth 5 | Out-File "runs\relay_exec\state.json" -Encoding utf8 }

function Test-GpuFree {
    $line = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if (-not $line) { return $false }
    $mem = [int]($line | Select-Object -First 1)
    if ($mem -ge 500) { return $false }
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "ntu_phaseb|run_p05|c1_decouple|warmstart|train_yolo|public_real" }
    return ($null -eq $procs)
}

function Wait-GpuFree {
    Write-Host "[gate] 等待 GPU 释放（每 5 分钟巡检，需持续空闲 10 分钟）..."
    $idleSince = $null
    while ($true) {
        $u = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null
        Write-Host ("[patrol] {0} -> {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $u)
        if (Test-GpuFree) {
            if (-not $idleSince) { $idleSince = Get-Date; Write-Host "[gate] 首次检测到空闲，进入 10 分钟确认期" }
            if (((Get-Date) - $idleSince).TotalSeconds -ge 600) { Write-Host "[gate] 确认期通过，启动接力"; return }
        } else { $idleSince = $null }
        Start-Sleep -Seconds 300
    }
}

function Invoke-Step {
    param($Name, [scriptblock]$Cmd, [string[]]$VerifyPaths, [string[]]$CommitPaths, [string]$CommitMsg)
    Write-Host "`n========== [$Name] START =========="
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        try { & $Cmd } catch { Write-Host "[${Name}] 异常: $_" }
        $code = $LASTEXITCODE
        $missing = @($VerifyPaths | Where-Object { -not (Test-Path $_) })
        if ($code -eq 0 -and $missing.Count -eq 0) {
            Write-Host "[$Name] OK（第 $attempt 次尝试）"
            git add @CommitPaths 2>&1 | Out-Null
            git commit -m $CommitMsg 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { Write-Host "[$Name][warn] commit 退出码 $LASTEXITCODE（可能无变更），继续" }
            $state.steps += @{ name = $Name; status = "done"; at = (Get-Date -Format o); attempts = $attempt }
            Save-State
            return
        }
        Write-Host "[$Name] 第 $attempt 次失败: exit=$code missing=$($missing -join ',')"
        if ($attempt -eq 1) { Start-Sleep -Seconds 120 }
    }
    $state.steps += @{ name = $Name; status = "HALT"; at = (Get-Date -Format o) }
    Save-State
    @{ halted_at = (Get-Date -Format o); step = $Name; note = "重试后仍失败，留证待人工" } |
        ConvertTo-Json | Out-File "runs\relay_exec\HALT.json" -Encoding utf8
    Write-Host "[$Name] HALT —— 停止后续队列，证据已留存"
    Stop-Transcript
    exit 1
}

New-Item -ItemType Directory -Force -Path "runs\relay_exec" | Out-Null
Save-State

Wait-GpuFree

# Q1 — AL full-budget 复跑
Invoke-Step "Q1_al_full" `
    { & $PY scripts\run_p05_al_efficiency.py --config configs\p05_al_full.yaml --fresh } `
    -VerifyPaths @("reports\p05-al-efficiency-full-$DATE.json") `
    -CommitPaths @("reports\p05-al-efficiency-full-$DATE.json", "configs\p05_al_full.yaml") `
    -CommitMsg "feat(p05): [relay Q1] AL full-budget 复跑归档——GPU 接力自动执行"

# Q2 — C1 解耦成本 full 档
Invoke-Step "Q2_c1_full" `
    { & $PY scripts\run_c1_decouple.py --tier full --device auto --output-json "reports\c1-decouple-cost-full-$DATE.json" } `
    -VerifyPaths @("reports\c1-decouple-cost-full-$DATE.json") `
    -CommitPaths @("reports\c1-decouple-cost-full-$DATE.json", "reports\c1-decouple-cost-$DATE.md") `
    -CommitMsg "feat(p06): [relay Q2] C1 解耦成本 full 档——GPU 接力自动执行；若与 small 档矛盾以 full 为准回改结论"

# Q3a — YOLO dog-pose 权重训练
Invoke-Step "Q3a_yolo_dogpose" `
    { & $PY scripts\train_yolo_dogpose.py --epochs 50 --batch 16 } `
    -VerifyPaths @("runs\public_real_yolo_dogpose\train\weights\best.pt") `
    -CommitPaths @("runs\public_real_yolo_dogpose") `
    -CommitMsg "feat(p05): [relay Q3a] YOLO dog-pose 24 点犬类权重训练完成——GPU 接力自动执行"

# Q3b — AK 全量提点
Invoke-Step "Q3b_extract" `
    { & $PY scripts\run_p05_public_real_pipeline.py --stage extract --weights "runs\public_real_yolo_dogpose\train\weights\best.pt" } `
    -VerifyPaths @("runs\public_real_dataset\partialclass4_T30.pkl", "runs\public_real_dataset\partialclass4_manifest.json") `
    -CommitPaths @("runs\public_real_dataset") `
    -CommitMsg "feat(p05): [relay Q3b] AK 172 视频全量提点完成——partialclass4_T30.pkl 落盘"

# Q3c — 公开真实层微调
Invoke-Step "Q3c_finetune" `
    { & $PY scripts\run_p05_public_real_finetune.py --pkl "runs\public_real_dataset\partialclass4_T30.pkl" --init "runs\p05_stgcn_bc_full\best.pt" --output-json "reports\p05-public-real-partialclass-result-$DATE.json" } `
    -VerifyPaths @("reports\p05-public-real-partialclass-result-$DATE.json") `
    -CommitPaths @("reports\p05-public-real-partialclass-result-$DATE.json", "runs\public_real_finetune") `
    -CommitMsg "feat(p05): [relay Q3c] 公开真实层 4 类微调完成——tab2 中间列数字落地（论文回填解锁）"

$state.finished_at = (Get-Date -Format o); $state.status = "ALL_DONE"; Save-State
Write-Host "`n[RELAY] 全部五步完成 ✅ —— tab2 公开真实层数字已落地，论文终稿回填解锁"
Stop-Transcript
