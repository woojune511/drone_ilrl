param(
    [string]$PythonVersion = "3.10",
    [switch]$SkipIL,
    [switch]$SkipDev
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CacheDir = Join-Path $ProjectRoot ".uv-cache"
$PythonInstallDir = Join-Path $ProjectRoot ".uv-python"
$VenvDir = Join-Path $ProjectRoot ".venv"

New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $PythonInstallDir | Out-Null
$env:UV_CACHE_DIR = $CacheDir
$env:UV_PYTHON_INSTALL_DIR = $PythonInstallDir

Write-Host "Using uv cache: $CacheDir"
Write-Host "Using uv Python install dir: $PythonInstallDir"
Write-Host "Ensuring Python $PythonVersion is available..."
uv python install $PythonVersion

if (Test-Path (Join-Path $VenvDir "Scripts\\python.exe")) {
    Write-Host "Virtual environment already exists at $VenvDir"
}
else {
    Write-Host "Creating virtual environment at $VenvDir"
    uv venv --python $PythonVersion $VenvDir
}

$SyncArgs = @("sync", "--extra", "drones")
if (-not $SkipIL) {
    $SyncArgs += @("--extra", "il")
}
if (-not $SkipDev) {
    $SyncArgs += @("--extra", "dev")
}

Push-Location $ProjectRoot
try {
    Write-Host "Syncing dependencies: $($SyncArgs -join ' ')"
    & uv @SyncArgs

    Write-Host "Running smoke test..."
    uv run python scripts/check_env.py
}
finally {
    Pop-Location
}
