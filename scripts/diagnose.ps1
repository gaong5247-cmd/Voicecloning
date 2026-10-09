$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
python launcher.py --diagnose
if ($LASTEXITCODE -ne 0) { throw 'Diagnostics failed' }
