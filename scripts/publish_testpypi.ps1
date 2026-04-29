Param(
  [string]$RepositoryUrl = "https://test.pypi.org/legacy/"
)

$ErrorActionPreference = "Stop"

if (-not $env:TWINE_PASSWORD) {
  Write-Error "TWINE_PASSWORD is not set. Set it to your TestPyPI token (starts with pypi-...) and re-run."
  exit 1
}

Write-Host "Building package..." -ForegroundColor Cyan
python -m pip install --upgrade pip | Out-Null
python -m pip install --upgrade build twine | Out-Null

if (Test-Path dist) { Remove-Item -Recurse -Force dist }

python -m build

Write-Host "Uploading to TestPyPI..." -ForegroundColor Cyan
twine upload --repository-url $RepositoryUrl -u __token__ -p $env:TWINE_PASSWORD dist/*

Write-Host "Done." -ForegroundColor Green

