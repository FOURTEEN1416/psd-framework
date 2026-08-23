# P0.2 E-A 唤醒评估包装脚本（定时任务入口）
# 由计划任务在训练预计完成后触发；幂等可重入；禁止启动任何新训练。
# 任务书（人读版）: reports/p02-eA-wakeup-task.md
param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Repo      = 'D:\Desktop\psd-framework'
$TaskName  = '\OpenCode\p02-eA-eval-wakeup'
$Log       = Join-Path $Repo 'runs\p02_smq_eA\wakeup_eval.log'
$Py        = Join-Path $Repo '.venv\Scripts\python.exe'
$Epoch30   = Join-Path $Repo 'runs\p02_smq_eA\models\epoch-30.model'
$ConcatOut = 'reports/p02-smq-iou-eA-concat.json'
$SeedsOut  = 'reports/p02-smq-iou-eA-seeds.json'

Set-Location $Repo
function Log($msg) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg; $line | Tee-Object -FilePath $Log -Append }
if ($DryRun) { Log 'DRYRUN: 校验通过，退出'; exit 0 }

function Test-TrainRunning {
    $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -match 'train_smq_segmentation' }
    return [bool]$p
}

try {
    # ── 1) 等待训练完成：轮询最多 40 分钟（每 3 分钟） ──
    $deadline = (Get-Date).AddMinutes(40)
    while ($true) {
        $running = Test-TrainRunning
        $ckptReady = Test-Path $Epoch30
        if (-not $running -and $ckptReady) { Log '训练已完成且 epoch-30 就绪'; break }
        if (-not $running -and -not $ckptReady) {
            Log 'CRASH: 训练进程消失但 epoch-30 缺失——归档崩溃证据'
            $evi = Join-Path $Repo 'reports\p02-eA-crash-evidence.md'
            "# E-A 崩溃证据 $(Get-Date -Format F)`n`n## train_log.txt 尾部``n``````" | Out-File $evi -Encoding utf8
            Get-Content (Join-Path $Repo 'runs\p02_smq_eA\train_log.txt') -Tail 50 -ErrorAction SilentlyContinue | Out-File $evi -Append -Encoding utf8
            "`n## console_err.log 尾部`n``````" | Out-File $evi -Append -Encoding utf8
            Get-Content (Join-Path $Repo 'runs\p02_smq_eA\console_err.log') -Tail 50 -ErrorAction SilentlyContinue | Out-File $evi -Append -Encoding utf8
            git add $evi; git commit -m 'docs(wip): P0.2 E-A 训练崩溃证据归档（自动唤醒代理）'
            schtasks /Delete /TN $TaskName /F 2>$null
            exit 2
        }
        if ((Get-Date) -gt $deadline) { Log 'TIMEOUT: 等待 40 分钟仍未完成——保留任务下次人工检查'; exit 3 }
        Log ('等待中… 进程运行={0} epoch30={1}' -f $running, $ckptReady)
        Start-Sleep -Seconds 180
    }

    # ── 2) 幂等：已有结果则跳过评估 ──
    if (Test-Path (Join-Path $Repo $ConcatOut)) {
        Log '结果已存在（幂等命中），跳过评估'
    } else {
        # ── 3) 保护 v3 基线可视化 ──
        foreach ($i in 1, 2) {
            $old = "reports\p02-vis-episode$i.png"
            if (Test-Path (Join-Path $Repo $old)) {
                Move-Item (Join-Path $Repo $old) ("reports\p02-vis-v3baseline-episode$i.png") -Force
                Log "已保护旧可视化 → p02-vis-v3baseline-episode$i.png"
            }
        }

        # ── 4) 双口径评估 ──
        Log '评估开始：concat 口径'
        & $Py scripts\eval_smq_segmentation.py --config configs\p02_smq_eA.yaml --iou --vis `
            --ckpt runs\p02_smq_eA\models\epoch-30.model --gt-protocol concat --out $ConcatOut 2>&1 |
            ForEach-Object { "$_" } | Out-File $Log -Append -Encoding utf8
        Log '评估开始：seeds 口径（规则种子伪GT conf>=0.8 且 >=0.5s）'
        & $Py scripts\eval_smq_segmentation.py --config configs\p02_smq_eA.yaml --iou --vis `
            --ckpt runs\p02_smq_eA\models\epoch-30.model --gt-protocol seeds --out $SeedsOut 2>&1 |
            ForEach-Object { "$_" } | Out-File $Log -Append -Encoding utf8

        foreach ($i in 1, 2) {
            $new = "reports\p02-vis-episode$i.png"
            if (Test-Path (Join-Path $Repo $new)) {
                Move-Item (Join-Path $Repo $new) ("reports\p02-vis-eA-episode$i.png") -Force
            }
        }

        # ── 5) 码本复检 ──
        Log '码本复检（运动词直方图/latent cos）'
        & $Py scripts\diag_p02_motion_words.py --config configs\p02_smq_eA.yaml `
            --ckpt runs\p02_smq_eA\models\epoch-30.model --out reports/p02-diag-motionwords-eA.json 2>&1 |
            ForEach-Object { "$_" } | Out-File $Log -Append -Encoding utf8
    }

    # ── 6) 摘要提取（公开真实层口径） ──
    $summary = @()
    $summary += "# E-A 自动评估摘要 $(Get-Date -Format F)"
    $summary += '指标口径: 公开真实层(InterPet4D smal_npy)'
    foreach ($pair in @(@('concat 拼接协议', $ConcatOut), @('seeds 种子伪GT', $SeedsOut))) {
        $label, $path = $pair
        $f = Join-Path $Repo $path
        if (Test-Path $f) {
            $j = Get-Content $f -Raw | ConvertFrom-Json
            $ious = @($j.episodes | ForEach-Object { $_.mean_matched_iou })
            $mean = if ($ious.Count) { [math]::Round(($ious | Measure-Object -Average).Average, 4) } else { 'N/A' }
            $summary += "- ${label}: mean_matched_iou=$mean (episodes=$($j.episodes.Count), 详情见 $path)"
        } else { $summary += "- ${label}: 缺失!" }
    }
    $diagF = Join-Path $Repo 'reports\p02-diag-motionwords-eA.json'
    if (Test-Path $diagF) { $summary += "- 码本复检: 见 p02-diag-motionwords-eA.json（对照基线 latent cos=1.0 为坍缩）" }
    $sumPath = Join-Path $Repo 'reports\p02-eA-eval-summary.md'
    $summary | Out-File $sumPath -Encoding utf8
    Get-Content $sumPath | ForEach-Object { Log $_ }

    # ── 7) 提交白名单产物 ──
    git add reports/p02-smq-iou-eA-concat.json reports/p02-smq-iou-eA-seeds.json `
            reports/p02-diag-motionwords-eA.json reports/p02-eA-eval-summary.md `
            reports/p02-vis-eA-episode1.png reports/p02-vis-eA-episode2.png 2>$null
    git commit -m 'feat(wip): P0.2 E-A 双口径评估+码本复检自动归档（唤醒代理执行，判读待用户）'
    Log 'Git 提交完成'

    # ── 8) 自清理：一次性任务完成后删除自己 ──
    schtasks /Delete /TN $TaskName /F 2>$null
    Log 'DONE Status=success'
    exit 0
}
catch {
    Log "FAILED Status=failed 原因: $_"
    schtasks /Delete /TN $TaskName /F 2>$null
    exit 1
}
