# Set up / upgrade the alpaca_gpu_lab venv (torch cu121 + model layer).
# ASCII-only on purpose (PowerShell 5.1 mangles non-ASCII). Run from anywhere.
# Idempotent: safe to re-run on the existing data-layer venv.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not (Test-Path (Join-Path $repo ".venv"))) {
    uv venv --python 3.12
}
$py = Join-Path $repo ".venv\Scripts\python.exe"

# torch from the cu121 index; --native-tls for this box's TLS interception.
uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cu121 --native-tls
uv pip install --python $py -r requirements.txt --native-tls

& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
& $py -c "import xgboost, sklearn; print('xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"
