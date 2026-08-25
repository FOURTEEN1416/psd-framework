# window_checkin.ps1 — 窗口自助收编+卸窗协议（AGENTS.md §4 条款7）
# 任何窗口完成验收自检后可自助执行；安全门禁固化在脚本内，无需等协调者人肉中转。
# 用法:
#   pwsh scripts/window_checkin.ps1 -Name W27                          # 收编（测试门禁+领地扫描+master回归）
#   pwsh scripts/window_checkin.ps1 -Name W27 -Remove                  # 收编后卸窗（含数据汇聚提醒）
#   pwsh scripts/window_checkin.ps1 -Name W27 -Message "移交说明..."   # 收编时附看板公告
# 冲突处理: 合并遇冲突自动 abort 并指引（琐碎冲突留协调者，勿自行 force）

param(
    [Parameter(Mandatory=$true)][string]$Name,
    [switch]$Remove,
    [switch]$SkipTest,
    [string]$Message,
    [string]$Permit   # 协调者特批的领地豁免（逗号分隔前缀，如 "docs/paper/introduction.md"）；使用时必须在看板公告注明
)

$ErrorActionPreference = "Continue"
$MAIN = "D:\Desktop\psd-framework"
$BRANCH = "wt/$Name"
$WORKTREE = "D:\Desktop\psd-framework-$Name"
$PY = Join-Path $MAIN ".venv\Scripts\python.exe"

# ---- 门禁 0: 前置存在性
if (-not (Test-Path $WORKTREE)) { Write-Error "[checkin] worktree 不存在: $WORKTREE"; exit 1 }

# ---- 门禁 1: 领地扫描（禁触清单，命中即拒）
$FORBIDDEN = @("dev-docs/decisions/", "docs/paper/", "*ntu*", "scripts/relay_executor.ps1", "dev-docs/HANDOVER.md")
$changed = git -C $MAIN diff --name-only master..$BRANCH 2>$null
if (-not $changed) { Write-Host "[checkin] $BRANCH 无领先提交——视为已收编，跳过合并"; }
else {
    $hits = @($changed | Where-Object { $f = $_
        $banned = ($FORBIDDEN | Where-Object { $f -like $_ }).Count -gt 0
        if (-not $banned -or -not $Permit) { return $banned }
        $allowed = ($Permit -split "," | Where-Object { $f -like "$($_)*" -or $f -like $_ }).Count -gt 0
        return ($banned -and -not $allowed) })
    if ($hits.Count -gt 0) {
        Write-Error "[checkin] 领地违规，拒绝收编。禁触文件: $($hits -join ', ') —— 如确需变更请走协调者特批"
        exit 1
    }
    Write-Host "[checkin] 领地扫描通过（$($changed.Count) 文件）"

    # ---- 门禁 2: 窗口内测试
    if (-not $SkipTest) {
        Write-Host "[checkin] 运行窗口内测试..."
        Push-Location $WORKTREE
        & $PY -m pytest psd -q 2>&1 | Select-Object -Last 3 | ForEach-Object { Write-Host "  $_" }
        $testExit = $LASTEXITCODE
        Pop-Location
        if ($testExit -ne 0) { Write-Error "[checkin] 窗口内测试未通过（exit=$testExit），拒绝收编"; exit 1 }
    } else { Write-Host "[checkin] 跳过测试（-SkipTest）" }

    # ---- 收编: --no-ff 强制合并节点
    Push-Location $MAIN
    git merge --no-ff $BRANCH -m "merge($Name): 自助收编——window_checkin 协议执行（测试门禁+领地扫描通过）" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        git merge --abort 2>$null
        Pop-Location
        Write-Error "[checkin] 合并冲突已自动 abort——琐碎冲突请留协调者裁决，或重基后重试"
        exit 1
    }
    # ---- master 回归守卫
    Write-Host "[checkin] master 全仓回归..."
    & $PY -m pytest psd -q 2>&1 | Select-Object -Last 2 | ForEach-Object { Write-Host "  $_" }
    if ($LASTEXITCODE -ne 0) {
        git reset --hard HEAD@{1} | Out-Null
        Pop-Location
        Write-Error "[checkin] master 回归失败，合并已回滚——修复后重试"
        exit 1
    }
    Pop-Location
    Write-Host "[checkin] ✅ $Name 已收编 master（--no-ff 节点保留）"
}

# ---- 数据汇聚提醒（runs/data_campaign 类产物不随 git）
$campaignSrc = Join-Path $WORKTREE "runs\data_campaign"
if (Test-Path $campaignSrc) {
    $dst = Join-Path $MAIN "runs\data_campaign"
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    robocopy $campaignSrc $dst /E /NFL /NDL /NJH /NJS /NP | Out-Null
    Write-Host "[checkin] 数据汇聚: runs/data_campaign → 主检出（robocopy 完成）"
}

# ---- 卸窗
if ($Remove) {
    Write-Host "[checkin] 卸窗: new_window_worktree.ps1 -Name $Name -Remove"
    pwsh (Join-Path $MAIN "scripts\new_window_worktree.ps1") -Name $Name -Remove 2>&1 | Select-Object -Last 4 | ForEach-Object { Write-Host "  $_" }
}

# ---- 看板公告
$boardMsg = if ($Message) { $Message } else { "已自助收编$(if ($Remove) {' 并卸窗'})" }
pwsh (Join-Path $MAIN "scripts\window_board.ps1") -Append ("[{0}][_{1}_] (ALL) HANDOFF: {2}" -f (Get-Date -Format 'MM-dd HH:mm'), $Name, $boardMsg)

Write-Host "[checkin] 完成。"
