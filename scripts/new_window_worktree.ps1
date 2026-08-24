#Requires -Version 7
<#
.SYNOPSIS
    W24 一键建窗（用户裁决 B-full）：为新协作窗口创建 git worktree 物理隔离环境。

.DESCRIPTION
    行为（W24 任务书 D1）:
      1. git worktree add ..\psd-framework-<Name> -b wt/<Name>
      2. data/ 目录 Junction 联接回主仓（gitignore 生成物不随 worktree 走；Junction 无需管理员权限）
      3. runs/ 独立新建 + 上游 checkpoint 按清单复制
      4. .venv 不复制——统一用主仓绝对解释器
      5. 打印该窗口的标准启动提示词与白名单提醒

    卸窗（收编完成后清理）: 加 -Remove 开关。
      ⚠️ 禁止直接删目录或裸跑 git worktree remove——git 会跟随 data Junction
      删到主仓内容（2026-08-25 冒烟实测误伤 data/.gitkeep）。本脚本会先摘
      Junction 再删树；分支默认仅在其已并入当前 HEAD 时才删除（未合并需 -ForceBranch）。

.PARAMETER Name
    窗口名（如 w25-al-full、_smoke）。允许字母/数字/下划线/连字符，须以字母、数字或下划线开头；
    分支名为 wt/<Name>，目录为主仓同级 ..\psd-framework-<Name>。

.PARAMETER Remove
    安全卸窗：摘 data Junction → git worktree remove → prune → 视合并状态删分支。

.PARAMETER ForceBranch
    仅 -Remove 生效：分支未并入时也强制删除（丢弃未收编提交，慎用）。

.EXAMPLE
    pwsh scripts/new_window_worktree.ps1 -Name w25-demo          # 建窗
    pwsh scripts/new_window_worktree.ps1 -Name w25-demo -Remove  # 收编后卸窗
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Name,

    [switch]$Remove,

    [switch]$ForceBranch
)

$ErrorActionPreference = "Stop"

# ── 0. 定位主仓根（以脚本自身位置为基准，不依赖 cwd —— 任务书风险预案要求）──
$MainRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ($Name -notmatch '^[A-Za-z0-9_][A-Za-z0-9_-]*$') {
    throw "窗口名 '$Name' 非法：只允许字母/数字/下划线/连字符，且须字母、数字或下划线开头"
}

$Branch      = "wt/$Name"
$ParentDir   = Split-Path $MainRoot -Parent
$WorktreeDir = Join-Path $ParentDir "psd-framework-$Name"

# ══════════════ 卸窗分支（-Remove）══════════════
if ($Remove) {
    Write-Host "[W24 卸窗] Name=$Name  Branch=$Branch"
    if (-not (Test-Path -LiteralPath $WorktreeDir)) {
        # 目录已不在：只做元数据清理
        git -C $MainRoot worktree prune
        Write-Host "[W24 卸窗] 目录不存在，已执行 worktree prune"
        return
    }
    if (-not ((Get-Item -LiteralPath $WorktreeDir -Force).FullName -like "$ParentDir\psd-framework-$Name")) {
        throw "路径校验失败，中止: $WorktreeDir"
    }

    # 1. 先摘 data Junction（必须先于任何 git 删除操作）
    $wtData = Join-Path $WorktreeDir "data"
    if (Test-Path -LiteralPath $wtData) {
        $item = Get-Item -LiteralPath $wtData -Force
        if ($item.LinkType) {
            if ($item.Target -and (([string]$item.Target) -notlike "$MainRoot*")) {
                throw "data Junction 指向非主仓($($item.Target))，中止以防误伤外部数据"
            }
            $item.Delete()
            Write-Host "[W24 卸窗] 已摘除 data Junction（主仓内容不受影响）"
        }
        else {
            Write-Warning "[W24 卸窗] data/ 是普通目录（可能为退化复制的分叉快照），保留原样待人工处置: $wtData"
        }
    }

    # 2. 检查未收编提交，防误删工作
    git -C $MainRoot merge-base --is-ancestor $Branch HEAD 2>$null
    $merged = ($LASTEXITCODE -eq 0)
    if (-not $merged) {
        Write-Warning "[W24 卸窗] 分支 $Branch 尚未并入当前 HEAD——存在未收编提交!"
        if (-not $ForceBranch) {
            Write-Warning "[W24 卸窗] 保留目录与分支；确认已收编后加 -ForceBranch 重跑，或先行 merge"
            git -C $MainRoot worktree remove --force $WorktreeDir
            git -C $MainRoot worktree prune
            Write-Host "[W24 卸窗] worktree 已移除，分支 $Branch 保留"
            return
        }
    }

    # 3. 删树 + 元数据 + 分支
    git -C $MainRoot worktree remove --force $WorktreeDir
    if ($LASTEXITCODE -ne 0) { throw "git worktree remove 失败" }
    git -C $MainRoot worktree prune
    git -C $MainRoot branch -D $Branch
    if ($LASTEXITCODE -ne 0) { throw "删除分支失败" }
    Write-Host "[W24 卸窗] ✅ 完成: 目录/元数据/分支均已清除" -ForegroundColor Green
    return
}

Write-Host ("=" * 64)
Write-Host "[W24 建窗] Name=$Name  Branch=$Branch"
Write-Host "[W24 建窗] 主仓: $MainRoot"
Write-Host "[W24 建窗] 目标: $WorktreeDir"
Write-Host ("=" * 64)

# 防重复：worktree 目录 / 分支任一已存在即拒绝
if (Test-Path -LiteralPath $WorktreeDir) {
    throw "目录已存在，拒绝覆盖: $WorktreeDir （如为废弃残留请人工确认后清理）"
}
git -C $MainRoot show-ref --verify --quiet "refs/heads/$Branch"
if ($LASTEXITCODE -eq 0) {
    throw "分支已存在，拒绝覆盖: $Branch"
}

# 已知必需的上游 checkpoint 清单（后续任务书可增补；相对主仓根路径）
$UpstreamCheckpoints = @(
    "runs/p05_stgcn_bc_full/best.pt"
)

function Copy-UpstreamCheckpoints {
    foreach ($rel in $UpstreamCheckpoints) {
        $src = Join-Path $MainRoot $rel
        $dst = Join-Path $WorktreeDir $rel
        if (-not (Test-Path -LiteralPath $src)) {
            Write-Warning "[ckpt] 上游 checkpoint 缺失，跳过: $src"
            continue
        }
        New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
        Copy-Item -LiteralPath $src -Destination $dst -Force
        Write-Host "[ckpt] 已复制: $rel"
    }
}

function New-DataJunction {
    $wtData    = Join-Path $WorktreeDir "data"
    $mainData  = Join-Path $MainRoot "data"
    # worktree checkout 后 data/ 仅含 git 跟踪的 .gitkeep，先移除再建 Junction
    if (Test-Path -LiteralPath $wtData) {
        $item = Get-Item -LiteralPath $wtData -Force
        if ($item.LinkType) { throw "意外状态: data 已是链接，中止以防误伤主仓" }
        Remove-Item -LiteralPath $wtData -Recurse -Force
    }
    try {
        New-Item -ItemType Junction -Path $wtData -Target $mainData | Out-Null
        Write-Host "[data] Junction 建立: $wtData -> $mainData"
    }
    catch {
        # 风险预案：Junction 失败退化为一次性镜像复制 + 数据分叉戳登记
        Write-Warning "[data] Junction 创建失败($($_.Exception.Message))，退化为 robocopy /MIR 一次性复制（非共享，存在分叉可能）"
        robocopy $mainData $wtData /MIR /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy 复制失败，退出码 $LASTEXITCODE" }
        $stamp = [ordered]@{
            forked     = $true
            created_at = (Get-Date).ToString("s")
            source     = $mainData
            note       = "Junction 不可用，本窗口 data 为一次性快照，与主仓存在分叉可能；生成物勿直接回写主仓同名路径"
        }
        $stamp | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $wtData "_FORK_STAMP.json") -Encoding utf8
        Write-Warning "[data] 已写入分叉戳: data\_FORK_STAMP.json"
    }
}

# ── 1. 建 worktree ──
git -C $MainRoot worktree add $WorktreeDir -b $Branch
if ($LASTEXITCODE -ne 0) { throw "git worktree add 失败" }

try {
    # ── 2. data/ Junction（失败自动退化复制）──
    New-DataJunction

    # ── 3. runs/ 独立新建（checkout 自带 runs/.gitkeep）+ checkpoint 清单复制 ──
    New-Item -ItemType Directory -Force -Path (Join-Path $WorktreeDir "runs") | Out-Null
    Copy-UpstreamCheckpoints

    # ── 4. .venv 不复制（文档约定见下方提示词）──
}
catch {
    Write-Warning "[rollback] 建窗失败，回滚 worktree 与分支..."
    git -C $MainRoot worktree remove --force $WorktreeDir 2>$null
    git -C $MainRoot branch -D $Branch 2>$null
    throw
}

# ── 5. 标准启动提示词与白名单提醒 ──
$PyExe = Join-Path $MainRoot ".venv\Scripts\python.exe"
Write-Host ""
Write-Host ("=" * 64)
Write-Host "✅ 窗口 [$Name] 就绪"
Write-Host ("=" * 64)
Write-Host @"
【新窗口启动提示词——复制给新会话】
你接手 psd-framework 窗口 $Name（worktree 物理隔离，B-full 协议）。
- 工作目录: $WorktreeDir （一切读写只在此目录内）
- 分支: $Branch （提交落在本分支，禁止切到 master 直接改）
- 收编: 完成后由协调窗口在主检出显式 git merge --no-ff $Branch
- Python 解释器（绝对路径，勿自建 venv）:
    "$PyExe"
  示例: & "$PyExe" -m pytest psd -q   （cwd 置于本 worktree 内即可 import psd）
- data/: 经 Junction 共享主仓数据（只读为主）；生成物带窗口前缀或唯一 seed
- runs/: 本窗独立；上游 checkpoint 已按清单复制
- 白名单纪律照旧: 只提交任务书白名单内文件，精确 git add，禁用 git add .
- 写 tracked 文件前必 git diff 重读；产物落盘即提交
- 卸窗: 收编完成后用
    pwsh "$MainRoot\scripts\new_window_worktree.ps1" $Name -Remove
  ⚠️ 禁止直接删目录或裸跑 git worktree remove（git 会跟随 data Junction 误删主仓内容）
"@ -ForegroundColor Cyan

