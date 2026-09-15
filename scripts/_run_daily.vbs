' Hidden launcher for run_daily.ps1 (window style 0). Norton blocks pythonw.exe
' creation in venvs, so python.exe runs hidden via wscript instead (qlib/hq pattern).
Set shell = CreateObject("WScript.Shell")
scriptDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File """ & scriptDir & "run_daily.ps1"""
shell.Run cmd, 0, False
