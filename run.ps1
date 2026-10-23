# ============================================================================
# Amharic Handwriting OCR — Windows PowerShell run script
# ============================================================================
# Run from the project root (after .\setup.ps1):
#
#   .\run.ps1
#
# Starts the FastAPI backend with uvicorn and serves the frontend SPA.
# The app becomes available at http://127.0.0.1:8000
#
# Requires an activated virtual environment. This script activates .venv if it
# exists. It does NOT assume `python` is on PATH outside the venv.
# ============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path $PSScriptRoot
Set-Location $ProjectRoot

$venvDir = Join-Path $ProjectRoot ".venv"
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"

# Use an existing activation, otherwise fall back to `python` / `py -3` directly.
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment ..." -ForegroundColor Cyan
    & $activateScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Could not activate virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "No .venv found. Run .\setup.ps1 first, or ensure `python` is on PATH." -ForegroundColor Yellow
}

# Verify the backend module can be imported before binding the port.
Write-Host "Verifying backend import ..." -ForegroundColor Cyan
$check = python -c "import src.api.app; print('import ok')" 2>$null
if (-not $check) {
    Write-Host "WARNING: backend import check failed (dependencies may be missing)." -ForegroundColor Yellow
}

$serverHost = "127.0.0.1"
$port = 8000
Write-Host "Starting Amharic OCR API at http://$serverHost`:$port ..." -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop.`n" -ForegroundColor Gray

python -m uvicorn src.api.app:app --host $serverHost --port $port --reload