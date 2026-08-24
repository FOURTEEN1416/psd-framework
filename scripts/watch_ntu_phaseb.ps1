# W9 NTU Phase B 训练看护（每30分钟由计划任务 \OpenCode\w9-ntu-watchdog 调用）
# 职责：只读巡检 + 唯一熔断（连续3个 Done 行 loss=nan → 终止训练并留证上报）
# 禁令：不改代码/配置、不做 git、不碰其他窗口领地（p02/p03/p05/supervisor）、除熔断外不杀任何进程
$ErrorActionPreference = "SilentlyContinue"
$root   = "D:\Desktop\psd-framework"
$log    = Join-Path $root "runs\ntu_phaseB\joint_pretext\log.txt"
$errLog = Join-Path $root "runs\ntu_phaseB\console_err.log"
$wdLog  = Join-Path $root "runs\ntu_phaseB\watchdog-log.md"
$ts     = Get-Date -Format "MM-dd HH:mm"

$proc = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
          Where-Object { $_.CommandLine -like "*run_ntu_phaseb*" })

# --- 唯一熔断：连续 >=3 个 Done 行 loss=nan（任务书预注册阈值）---
if ((Test-Path $log) -and $proc.Count -gt 0) {
  $done = @(Select-String -Path $log -Pattern "Done\.") | Select-Object -Last 12
  $streak = 0
  for ($i = $done.Count - 1; $i -ge 0; $i--) {
    if ($done[$i].Line -match "loss:\s*(nan|NAN|NaN)") { $streak++ } else { break }
  }
  if ($streak -ge 3) {
    $proc | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    $evidence = @(
      "# NTU Phase B 熔断证据", "",
      "- 触发时间: $(Get-Date -Format o)",
      "- 熔断依据: log.txt 连续 $streak 个 Done 行 loss=nan（预注册阈值=3）",
      "- 已终止进程 PID: $($proc[0].ProcessId)", "",
      "## log.txt 尾部50行", "",
      '```',
      (Get-Content $log -Tail 50 | Out-String),
      '```'
    )
    New-Item -ItemType Directory -Force -Path (Join-Path $root "reports") | Out-Null
    $evidence | Out-File (Join-Path $root "reports\ntu-phaseB-crash-evidence.md") -Encoding UTF8
    Add-Content $wdLog "[$ts] WARNING NaN熔断触发——已终止 PID=$($proc[0].ProcessId)，证据见 reports/ntu-phaseB-crash-evidence.md，待用户裁决"
    exit 0
  }
}

# --- 常规简报 / 异常退出判读 ---
$rawMem = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
$mem = 0; if ($rawMem) { $mem = [int]($rawMem -replace '[^0-9]', '') }
if ($proc.Count -gt 0) {
  $last = ""
  if (Test-Path $log) { $last = (Get-Content $log -Tail 1) -join " " }
  $risk = ""; if ($mem -ge 8000) { $risk = " WARN接近OOM边界" }
  Add-Content $wdLog "[$ts] alive=yes pid=$($proc[0].ProcessId) mem=${mem}MiB$risk | $last"
} else {
  $ageMin = -1
  if (Test-Path $log) {
    $ageMin = [int]((Get-Date) - (Get-Item $log).LastWriteTime).TotalMinutes
  }
  $errTail = ""
  if (Test-Path $errLog) { $errTail = (Get-Content $errLog -Tail 5) -join " | " }
  $kind = "unknown"
  if ($errTail -match "out of memory") { $kind = "resource-OOM" }
  elseif ($errTail -match "Traceback|Error") { $kind = "suspect-code-bug" }
  elseif ($ageMin -ge 0 -and $ageMin -lt 35) { $kind = "maybe-normal-exit" }
  Add-Content $wdLog "[$ts] alive=no logSilent=${ageMin}min verdict=$kind errTail=[[$errTail]] <-- 需人工复核"
}
