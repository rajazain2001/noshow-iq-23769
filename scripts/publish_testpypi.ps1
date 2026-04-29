Param(
  [string]$RepositoryUrl = "https://test.pypi.org/legacy/"
)

$ErrorActionPreference = "Stop"

if (-not $env:TWINE_PASSWORD) {
  Write-Error "TWINE_PASSWORD is not set. Set it to your TestPyPI token (starts with pypi-...) and re-run."
  exit 1
}

$py = Join-Path (Join-Path $PSScriptRoot "..") ".venv\\Scripts\\python.exe"
if (-not (Test-Path $py)) {
  Write-Error "Virtualenv python not found at $py. Create .venv first, then re-run."
  exit 1
}

Write-Host "Building package..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip | Out-Null
& $py -m pip install --upgrade build twine | Out-Null

if (Test-Path dist) { Remove-Item -Recurse -Force dist }

& $py -m build

Write-Host "Uploading to TestPyPI..." -ForegroundColor Cyan
& $py -m twine upload --repository-url $RepositoryUrl -u __token__ -p $env:TWINE_PASSWORD dist/*

Write-Host "Done." -ForegroundColor Green

