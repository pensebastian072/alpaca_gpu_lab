# Phase-2 battery matrix driver. ASCII-only. Sequential + fault-tolerant:
# each battery writes its own scorecard, so a crash mid-run loses only the
# in-flight battery (re-run resumes — registry is append-only, features cached).
# Order: small timeframes first (fast wins land before the heavy 1Min/5Min).
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
$feat = Join-Path $repo "market_data\features"

$tfs = @("4Hour", "1Hour", "15Min", "5Min", "1Min")
$proxies = @("IEF", "TLT", "VIXY")   # B08 cross-section needs the full core-7

foreach ($tf in $tfs) {
    # ensure the 3 proxy feature parquets exist at this timeframe (targets already built)
    foreach ($p in $proxies) {
        $fp = Join-Path $feat "${p}_${tf}.parquet"
        if (-not (Test-Path $fp)) {
            Write-Host "[build] $p $tf"
            & $py -u -m src.features.build --symbol $p --timeframe $tf --cross --enrich 2>$null
        }
    }
    # B06 direction_rank (1Min already done in a prior run)
    if ($tf -ne "1Min") {
        Write-Host "[B06] $tf"; & $py -u -m src.models.direction_rank --timeframe $tf
    }
    Write-Host "[B07] $tf"; & $py -u -m src.models.magnitude --timeframe $tf
    Write-Host "[B08] $tf"; & $py -u -m src.models.xsect --timeframe $tf
}
Write-Host "MATRIX DONE"
