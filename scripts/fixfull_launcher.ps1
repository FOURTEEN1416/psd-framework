# fixfull_trans002 launcher (SYSTEM scheduled task entry)
# NOTE: ASCII only - PowerShell 5.1 mangles UTF-8 Chinese comments and swallows the next line
# Session-terminated child processes die with the session (2026-09-08 13:49 incident) -
# always launch via schtasks SYSTEM context: PSD_TRANS002_FixFull
$log = "D:\Desktop\psd-framework\reports\fixfull_bootstrap.log"
"[$(Get-Date -Format s)] launcher start" | Out-File $log -Append -Encoding utf8
Set-Location D:\Desktop\psd-framework
& D:\Desktop\psd-framework\.venv\Scripts\python.exe D:\Desktop\psd-framework\scripts\fixfull_trans002.py *>> "D:\Desktop\psd-framework\reports\fixfull_console.log"
"[$(Get-Date -Format s)] launcher exit code=$LASTEXITCODE" | Out-File $log -Append -Encoding utf8
