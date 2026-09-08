# fixfull_trans002 启动器（SYSTEM 计划任务入口）
# 会话子进程会被会话终结连坐杀死(2026-09-08 13:49 事故)——必须经 schtasks SYSTEM 上下文脱离会话存活
$log = "D:\Desktop\psd-framework\reports\fixfull_bootstrap.log"
"[$(Get-Date -Format s)] launcher start" | Out-File $log -Append -Encoding utf8
Set-Location D:\Desktop\psd-framework
& D:\Desktop\psd-framework\.venv\Scripts\python.exe D:\Desktop\psd-framework\scripts\fixfull_trans002.py *>> "D:\Desktop\psd-framework\reports\fixfull_console.log"
"[$(Get-Date -Format s)] launcher exit code=$LASTEXITCODE" | Out-File $log -Append -Encoding utf8
