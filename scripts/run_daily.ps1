# AlpacaGpuDaily task body. ASCII-only (PowerShell 5.1).
# Start-Process with explicit stream redirect: the uv python trampoline hangs
# under Task Scheduler when streams are inherited (skill_forge lesson).
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
$logDir = Join-Path $repo "journal\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$out = Join-Path $logDir "daily_$stamp.log"
$err = Join-Path $logDir "daily_$stamp.err.log"

# PREFLIGHT: a missing interpreter used to fail SILENTLY. Start-Process threw, the
# $ErrorActionPreference="Stop" killed this script, and the task's .vbs launcher still
# exited 0 - so AlpacaGpuDaily reported success while writing 0-byte logs for 9 days
# (2026-09-01..09-08, after the repo moved to D: without its venv being rebuilt).
# Write the reason INTO the error log so it is visible to anything that reads output.
if (-not (Test-Path $py)) {
    "FATAL $(Get-Date -Format s): interpreter missing at $py - venv not built. Rebuild with perf_probe\migrationebuild_torch_venvs.ps1" |
        Out-File -FilePath $err -Encoding ascii
    exit 1
}

$proc = Start-Process -FilePath $py -ArgumentList "-m","src.daily","--once" `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err
$proc | Wait-Process -Timeout 5400 -ErrorAction SilentlyContinue
if (-not $proc.HasExited) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Add-Content $err "KILLED: exceeded 90 minute timeout"
}
