# P1 Autopilot wrapper - run by scheduled task every 15 min
$ErrorActionPreference = "Continue"
Set-Location "D:\Desktop\PR论文\psd-framework"
& ".\.venv\Scripts\python.exe" ".\scripts\p1_sprint\autopilot.py" 2>&1 | Out-File -FilePath "D:\Desktop\PR论文\psd-framework\runs\p1_sprint\autopilot_run.log" -Append -Encoding utf8
