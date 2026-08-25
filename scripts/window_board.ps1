# window_board.ps1 — 跨窗口消息看板读写工具
# 用法:
#   pwsh scripts/window_board.ps1 -Append "[12-40][_W25_] (ALL) INFO: xxx"   # 追加
#   pwsh scripts/window_board.ps1 -Tail 20                                   # 看最近 20 行
# 说明: BOARD 固定写在主检出绝对路径（所有 worktree 同盘可达），追加模式并发安全(重试3次)

param(
    [string]$Append,
    [int]$Tail = 0,
    [switch]$All
)

$BOARD = "D:\Desktop\psd-framework\dev-docs\board\BOARD.md"

if ($Append) {
    $ok = $false
    for ($i = 0; $i -lt 3 -and -not $ok; $i++) {
        try {
            Add-Content -LiteralPath $BOARD -Value $Append -Encoding UTF8 -ErrorAction Stop
            $ok = $true
        } catch { Start-Sleep -Milliseconds (200 * ($i + 1)) }
    }
    if ($ok) { Write-Host "[board] 已追加: $Append" }
    else { Write-Error "[board] 追加失败（重试 3 次）——请改为手动编辑或稍后再试"; exit 1 }
    exit 0
}

$lines = Get-Content -LiteralPath $BOARD -Encoding UTF8
if ($All -or $Tail -le 0) { $lines | ForEach-Object { $_ } }
else { $lines | Select-Object -Last $Tail | ForEach-Object { $_ } }
