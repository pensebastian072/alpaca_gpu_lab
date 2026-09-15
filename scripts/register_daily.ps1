# Register the AlpacaGpuDaily scheduled task (LIMITED principal, no elevation).
# ASCII-only. Run once: powershell -ExecutionPolicy Bypass -File scripts\register_daily.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$vbs = Join-Path $repo "scripts\_run_daily.vbs"
$name = "AlpacaGpuDaily"

$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument """$vbs"""
$trigger = New-ScheduledTaskTrigger -Daily -At "17:45"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null
Write-Host "Registered task $name (daily 17:45, hidden via wscript)."
