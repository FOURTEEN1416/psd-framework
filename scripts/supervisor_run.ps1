# 夜间监督代理启动器（Windows 计划任务入口，绝对路径模式）
# 由计划任务 \OpenCode\psd-overnight-supervisor 每 15 分钟调用。
Set-Location -LiteralPath "D:\Desktop\psd-framework"
& opencode run -- "读 reports/supervisor-brief.md 并严格执行其中的每轮步骤与全部禁令。"
exit $LASTEXITCODE
