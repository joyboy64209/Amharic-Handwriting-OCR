# ============================================================================
# Amharic Handwriting OCR — Windows PowerShell setup
# ============================================================================
# Run once from the project root (C:\Projects\Amharic-Handwriting-OCR):
#
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned  # if scripts are blocked
#   .\setup.ps1
#
# This creates a virtual environment at .venv and installs dependencies from
# requirements.txt. It does NOT download datasets or models; those are expected
# to already exist (copied from the source clone): data/uni_dataset/ and
# models/amharic_ocr_v2.pth.
# ============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path $PSScriptRoot
Write-Host "Project root: $ProjectRoot" -ForegroundColor Cyan

# --- Locate a Python interpreter -------------------------------------------
# Prefer `python` (3.12 in this environment, has pip). Fall back to `py -3`.
function Find-Python {
    foreach ($cmd in @("python", "py -3")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) { return $cmd }
    }
    return $null
}

$pyCmd = Find-Python
if (-not $pyCmd) {
    Write-Host "ERROR: No Python interpreter found. Install Python and ensure `python` or `py` is on PATH." -ForegroundColor Red
    exit 1
}
Write-Host "Using Python: $pyCmd" -ForegroundColor Green

# Check that pip is available (py -3 / Python 3.15 on this machine has no pip).
$pipCheck = & $pyCmd -m pip --version 2>$null
if (-not $pipCheck) {
    Write-Host "WARNING: pip is not available for `$pyCmd." -ForegroundColor Yellow
    Write-Host "   If you are using `py -3` (Python 3.15), bootstrap pip first, or use the `python` interpreter (3.12) which includes pip." -ForegroundColor Yellow
    Write-Host "   Example: python -m venv .venv`n   .\venv\Scripts\Activate.ps1`n   python -m pip install -r requirements.txt" -ForegroundColor Yellow
}

# --- Create virtual environment --------------------------------------------
$venvDir = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment at $venvDir ..."
    & $pyCmd -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists at $venvDir."
}

# --- Activate and install dependencies --------------------------------------
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
Write-Host "Activating environment and installing dependencies..."
& $activateScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Could not activate the virtual environment." -ForegroundColor Red
    exit 1
}

$reqFile = Join-Path $ProjectRoot "requirements.txt"
if (Test-Path $reqFile) {
    & $pyCmd -m pip install --upgrade pip 2>$null | Out-Null
    & $pyCmd -m pip install -r $reqFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Dependency installation failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "Dependencies installed." -ForegroundColor Green
} else {
    Write-Host "WARNING: requirements.txt not found; skipping pip install." -ForegroundColor Yellow
}

# --- Verify expected data/model files ---------------------------------------
Write-Host "`nChecking expected files:" -ForegroundColor Cyan
$checks = @(
    @{ p = "data/uni_dataset"; d = "isolated-character dataset" },
    @{ p = "models/amharic_ocr_v2.pth"; d = "CNN checkpoint" }
)
foreach ($c in $checks) {
    $full = Join-Path $ProjectRoot $c.p
    if (Test-Path $full) {
        Write-Host "  [OK] $($c.d): $($c.p)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $($c.d): $($c.p) - see docs/DATASET.md and docs/LIMITS.md" -ForegroundColor Yellow
    }
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next: .\run.ps1   (starts the API and serves the frontend)"
Write-Host "Then open: http://127.0.0.1:8000"