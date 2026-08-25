# relay_executor.ps1 v2 — PSD-Framework GPU 接力总执行器
# v2 修复: ①GPU 门禁阈值适配桌面基线(壁纸/浏览器常驻 ~1.8-2.4GB,旧 <500MiB 永不触发)
#          ②步骤校验从存在性升级为内容级(W28 上报采纳: 样本量/字段/大小)
# 队列: Q1(AL full) → Q2(C1 full) → Q3a(YOLO权重) → Q3b(全量提点) → Q3c(公开真实层微调)

$ErrorActionPreference = "Continue"
Set-Location "D:\Desktop\psd-framework"
Start-Transcript -Path "runs\relay_exec\transcript.log" -Append | Out-Null

$PY = "D:\Desktop\psd-framework\.venv\Scripts\python.exe"
$DATE = "2026-08-25"
$state = @{ started_at = (Get-Date -Format o); version = "v2"; steps = @() }
function Save-State { $state | ConvertTo-Json -Depth 5 | Out-File "runs\relay_exec\state.json" -Encoding utf8 }

function Test-GpuFree {
    # 桌面基线 ~1.8-2.4GB(壁纸引擎/浏览器/OpenCode 常驻)——阈值 2600MB
    $line = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if (-not $line) { return $false }
    $mem = [int]($line | Select-Object -First 1)
    if ($mem -ge 2600) { return $false }
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "ntu_phaseb|run_p05|c1_decouple|warmstart|train_yolo|public_real" }
    return ($null -eq $procs)
}

function Wait-GpuFree {
    Write-Host "[gate] v2 门禁: 显存<2600MB(桌面基线+余量) 且无 psd 训练进程,持续 600s"
    $idleSince = $null
    while ($true) {
        $u = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null
        Write-Host ("[patrol] {0} -> {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $u)
        if (Test-GpuFree) {
            if (-not $idleSince) { $idleSince = Get-Date; Write-Host "[gate] 首次空闲,10 分钟确认期" }
            if (((Get-Date) - $idleSince).TotalSeconds -ge 600) { Write-Host "[gate] 通过,启动接力"; return }
        } else { $idleSince = $null }
        Start-Sleep -Seconds 300
    }
}

# 内容级校验器(W28 上报采纳): 返回 $true=校验通过
function Test-JsonField { param($Path, $Field)
    if (-not (Test-Path $Path)) { return $false }
    try { $j = Get-Content $Path -Raw | ConvertFrom-Json; return ($null -ne ($j.PSObject.Properties[$Field])) }
    catch { return $false }
}

function Invoke-Step {
    param($Name, [scriptblock]$Cmd, [scriptblock]$Verify, [string[]]$CommitPaths, [string]$CommitMsg)
    Write-Host "`n========== [$Name] START =========="
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        try { & $Cmd } catch { Write-Host "[${Name}] 异常: $_" }
        $code = $LASTEXITCODE
        $ok = & $Verify
        if ($code -eq 0 -and $ok) {
            Write-Host "[$Name] OK(含内容校验,第 $attempt 次)"
            git add @CommitPaths 2>&1 | Out-Null
            git commit -m $CommitMsg 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { Write-Host "[$Name][warn] commit 无变更,继续" }
            $state.steps += @{ name = $Name; status = "done"; at = (Get-Date -Format o); attempts = $attempt }
            Save-State
            return
        }
        Write-Host "[$Name] 第 $attempt 次失败: exit=$code verify=$ok"
        if ($attempt -eq 1) { Start-Sleep -Seconds 120 }
    }
    $state.steps += @{ name = $Name; status = "HALT"; at = (Get-Date -Format o) }
    Save-State
    @{ halted_at = (Get-Date -Format o); step = $Name } | ConvertTo-Json |
        Out-File "runs\relay_exec\HALT.json" -Encoding utf8
    Write-Host "[$Name] HALT —— 停链留证"
    Stop-Transcript
    exit 1
}

New-Item -ItemType Directory -Force -Path "runs\relay_exec" | Out-Null
Save-State
Wait-GpuFree

# Q1 — AL full-budget(内容校验: JSON 含 summary.best_val_acc)
Invoke-Step "Q1_al_full" `
    { & $PY scripts\run_p05_al_efficiency.py --config configs\p05_al_full.yaml --fresh } `
    -Verify { Test-JsonField "reports\p05-al-efficiency-full-$DATE.json" "summary" } `
    -CommitPaths @("reports\p05-al-efficiency-full-$DATE.json", "configs\p05_al_full.yaml") `
    -CommitMsg "feat(p05): [relay Q1] AL full-budget 复跑归档(GPU 接力 v2)"

# Q2 — C1 full 档(内容校验: JSON 含 aggregated)
Invoke-Step "Q2_c1_full" `
    { & $PY scripts\run_c1_decouple.py --tier full --device auto --output-json "reports\c1-decouple-cost-full-$DATE.json" } `
    -Verify { Test-JsonField "reports\c1-decouple-cost-full-$DATE.json" "aggregated" } `
    -CommitPaths @("reports\c1-decouple-cost-full-$DATE.json") `
    -CommitMsg "feat(p06): [relay Q2] C1 解耦成本 full 档(GPU 接力 v2);与 small 档矛盾以 full 为准"

# Q3a — YOLO 权重(内容校验: best.pt >10MB)
Invoke-Step "Q3a_yolo_dogpose" `
    { & $PY scripts\train_yolo_dogpose.py --epochs 50 --batch 16 } `
    -Verify { (Test-Path "runs\public_real_yolo_dogpose\train\weights\best.pt") -and ((Get-Item "runs\public_real_yolo_dogpose\train\weights\best.pt").Length -gt 10MB) } `
    -CommitPaths @("runs\public_real_yolo_dogpose") `
    -CommitMsg "feat(p05): [relay Q3a] YOLO dog-pose 24 点权重(GPU 接力 v2)"

# Q3b — 全量提点(内容校验: pkl>1MB 且质量 JSON n_samples>0)
Invoke-Step "Q3b_extract" `
    { & $PY scripts\run_p05_public_real_pipeline.py --stage extract --weights "runs\public_real_yolo_dogpose\train\weights\best.pt" } `
    -Verify { (Test-Path "runs\public_real_dataset\partialclass4_T30.pkl") -and ((Get-Item "runs\public_real_dataset\partialclass4_T30.pkl").Length -gt 1MB) -and (Test-JsonField "runs\public_real_dataset\partialclass4_extract_quality.json" "n_samples") } `
    -CommitPaths @("runs\public_real_dataset") `
    -CommitMsg "feat(p05): [relay Q3b] AK 全量提点完成(GPU 接力 v2)"

# Q3c — 公开真实层微调(内容校验: JSON 含 summary.best_val_acc)
Invoke-Step "Q3c_finetune" `
    { & $PY scripts\run_p05_public_real_finetune.py --pkl "runs\public_real_dataset\partialclass4_T30.pkl" --init "runs\p05_stgcn_bc_full\best.pt" --output-json "reports\p05-public-real-partialclass-result-$DATE.json" } `
    -Verify { Test-JsonField "reports\p05-public-real-partialclass-result-$DATE.json" "summary" } `
    -CommitPaths @("reports\p05-public-real-partialclass-result-$DATE.json", "runs\public_real_finetune") `
    -CommitMsg "feat(p05): [relay Q3c] 公开真实层微调完成——tab2 数字落地(GPU 接力 v2)"

$state.finished_at = (Get-Date -Format o); $state.status = "ALL_DONE"; Save-State
Write-Host "`n[RELAY v2] 五步全成 ✅ tab2 公开真实列落地"
Stop-Transcript
