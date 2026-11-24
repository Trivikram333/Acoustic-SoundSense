$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements first. See README.md.' }
$env:PYTHONPATH = Join-Path $PSScriptRoot 'backend\src'
& $python -m soundsense
exit $LASTEXITCODE
