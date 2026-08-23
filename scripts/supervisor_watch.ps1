# PSD 夜间机械监督（纯 PowerShell，无 LLM 依赖 · 每 15 分钟由计划任务调用）
# 职责: 检测 P0.2 冲刺新产出 → 对照预注册判据 → 追加日志 → 仅提交日志文件。
# 判据来源: dev-docs/decisions/0003-p02-sprint-and-metric-ruling.md
#   成功 = 种子伪GT mean IoU >= 0.45；停止规则 = 连续两个实验无改进（记录，不决策）。

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\Desktop\psd-framework"
$log = "reports\supervisor-log.md"
$stateFile = "data\supervisor.state"

function Log([string]$line) { Add-Content -Path $log -Value $line -Encoding utf8 }

if (-not (Test-Path $log)) {
    $head = (git rev-parse --short HEAD)
    Log "# Supervisor Log（机械监督，每 15 分钟）"
    Log "[init] 监督上线 @ $(Get-Date -Format 'MM-dd HH:mm')，基线 commit $head"
    Log ""
}

$lastProcessed = if (Test-Path $stateFile) { Get-Content $stateFile -Raw } else { "" }
$lastLogTime = if (Test-Path $log) { (Get-Item $log).LastWriteTime } else { [datetime]::MinValue }

# 候选产出: reports 下 p02 相关 json/md，晚于上次处理时间
$cands = Get-ChildItem reports -Filter "p02-*" -File |
    Where-Object { $_.LastWriteTime -gt $lastLogTime.AddSeconds(-1) -and $_.Name -ne "supervisor-log.md" } |
    Sort-Object LastWriteTime

$new = @($cands | Where-Object { "$($_.FullName)|$($_.LastWriteTime.Ticks)" -ne $lastProcessed })
$changed = $false

foreach ($f in $new) {
    $ts = Get-Date -Format "MM-dd HH:mm:ss"
    Log "## [$ts] 新产出: $($f.Name) (mtime $($f.LastWriteTime.ToString('HH:mm')))"
    if ($f.Extension -eq ".json" -and $f.Name -match "iou") {
        try {
            $j = Get-Content $f.FullName -Raw | ConvertFrom-Json
            $agg = $j.aggregate
            if ($null -ne $agg) {
                Log ("  mean_matched_iou={0} boundary_f1={1} n_episodes={2} protocol={3}" -f `
                    $agg.mean_matched_iou, $agg.boundary_f1_mean, $agg.n_episodes, $j.protocol)
                $iou = [double]$agg.mean_matched_iou
                if ($iou -ge 0.45) { Log "  >>> 判读: 达到冲刺成功线 0.45 ✅（晨会可讨论收口）" }
                elseif ($iou -ge 0.43) { Log "  >>> 判读: 接近成功线(>=0.43 备用线)，视 ep1 条件由人工确认" }
                else { Log ("  >>> 判读: 未达线（目标 0.45）；较上一结果 {0}" -f `
                    $(if ($script:lastIou) { "对比见上" } else { "为首见" })) }
            }
        } catch { Log "  [解析失败] $_" }
    }
    $changed = $true
    Set-Content $stateFile -Value "$($f.FullName)|$($f.LastWriteTime.Ticks)" -Encoding ascii
}

if (-not $changed) {
    # 心跳节流: 距上条日志 >2h 才写一行
    if (((Get-Date) - $lastLogTime).TotalHours -ge 2) {
        Log "[heartbeat] $(Get-Date -Format 'MM-dd HH:mm') 无新产出"
        $changed = $true
    }
}

if ($changed) {
    git add $log | Out-Null
    git commit -m ("chore(supervisor): 巡检 {0} —— 日志更新" -f (Get-Date -Format "MM-dd HH:mm")) | Out-Null
}
