param([ValidateSet('cpu','xpu')][string]$Backend='xpu',[switch]$SkipInstall)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Run-Python { & python @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $args" } }
function Mark-Stage([string]$Message) { Write-Host ("[{0}] {1}" -f (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), $Message) }
function Test-Frozen([string]$Argument) {
    $exe = 'dist/CloneVoiceStudio/CloneVoiceStudio.exe'
    $process = Start-Process $exe -ArgumentList $Argument -PassThru
    if (-not $process.WaitForExit(240000)) {
        try { $process.Kill($true) } catch {}
        throw "Frozen $Argument timed out after 240 seconds"
    }
    if ($process.ExitCode -ne 0) { throw "Frozen $Argument failed: $($process.ExitCode)" }
}
Mark-Stage "Starting $Backend portable build"
if (-not $SkipInstall) {
    Mark-Stage "Installing $Backend PyTorch and application dependencies"
    Run-Python -m pip install -r "requirements/$Backend.txt"
    Run-Python -m pip install -r requirements/base.txt -r requirements/studio.txt -r requirements/build.txt -c requirements/windows-lock.txt
}
Mark-Stage 'Checking installed packages and running tests'
Run-Python -m pip check
Run-Python -m pytest tests -q
$env:QT_QPA_PLATFORM='offscreen'
Run-Python launcher.py --smoke-test
Remove-Item Env:QT_QPA_PLATFORM
Mark-Stage 'Preparing FFmpeg'
New-Item -ItemType Directory -Force bin | Out-Null
if (-not (Test-Path bin/ffmpeg.exe)) {
    Invoke-WebRequest 'https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-essentials_build.zip' -OutFile ffmpeg.zip
    Expand-Archive ffmpeg.zip -DestinationPath ffmpeg-temp -Force
    Copy-Item (Get-ChildItem ffmpeg-temp -Filter ffmpeg.exe -Recurse | Select-Object -First 1).FullName bin/ffmpeg.exe
    New-Item -ItemType Directory -Force ffmpeg-notices | Out-Null
    Get-ChildItem ffmpeg-temp -Recurse -File | Where-Object { $_.Name -match 'LICENSE|README' } | Copy-Item -Destination ffmpeg-notices
    Remove-Item ffmpeg.zip
}
# PyInstaller's maintained torch hook already collects torch submodules and data.
# --collect-all torch repeats that expensive discovery and pulls unrelated files.
# Keep native torch DLLs explicitly for Windows Intel XPU runtime compatibility.
Mark-Stage 'Building EXE with PyInstaller (using built-in torch hook)'
Run-Python -m PyInstaller --noconfirm --clean --onedir --windowed --name CloneVoiceStudio --paths vendor/seed_vc `
    --add-data 'vendor/seed_vc;vendor/seed_vc' --add-data 'config;config' --add-binary 'bin/ffmpeg.exe;bin' `
    --collect-binaries torch --collect-all torchaudio --collect-all sounddevice --collect-all soundfile `
    --collect-all transformers --collect-all librosa --collect-all speechbrain --collect-all pyannote.audio `
    --collect-all torchcodec --hidden-import seed_loader --hidden-import hf_utils `
    --hidden-import modules.flow_matching --hidden-import modules.length_regulator `
    --hidden-import modules.campplus.DTDNN --hidden-import modules.bigvgan.bigvgan launcher.py
$env:QT_QPA_PLATFORM='offscreen'
Mark-Stage 'Testing packaged GUI (240s timeout)'
Test-Frozen '--smoke-test'
Remove-Item Env:QT_QPA_PLATFORM
Mark-Stage 'Testing packaged model imports (240s timeout)'
Test-Frozen '--engine-import-test'
if ($Backend -eq 'cpu') {
    Mark-Stage 'Running actual packaged CPU conversion smoke test'
    Run-Python scripts/frozen_model_smoke.py dist/CloneVoiceStudio/CloneVoiceStudio.exe validation-data
}
Copy-Item README.md,LICENSE,THIRD_PARTY_NOTICES.md dist/CloneVoiceStudio/
if (Test-Path ffmpeg-notices) { Copy-Item -Recurse ffmpeg-notices dist/CloneVoiceStudio/ }
Run-Python -m pip freeze --exclude-editable | Out-File -Encoding utf8 dist/CloneVoiceStudio/dependency-versions.txt
Copy-Item "$env:LOCALAPPDATA/CloneVoiceStudio/logs/app.log" dist/CloneVoiceStudio/ci-smoke.log -ErrorAction SilentlyContinue
Copy-Item -Recurse docs dist/CloneVoiceStudio/docs -Force
Copy-Item -Recurse vendor dist/CloneVoiceStudio/source-vendor -Force
Copy-Item -Recurse app dist/CloneVoiceStudio/source-app -Force
Copy-Item launcher.py dist/CloneVoiceStudio/
Copy-Item -Recurse scripts dist/CloneVoiceStudio/source-scripts -Force
Copy-Item -Recurse requirements dist/CloneVoiceStudio/source-requirements -Force
Copy-Item -Recurse config dist/CloneVoiceStudio/source-config -Force
if (Test-Path validation/frozen-model.json) { Copy-Item validation/frozen-model.json dist/CloneVoiceStudio/verification.json }
Copy-Item config/settings.template.json dist/CloneVoiceStudio/settings.template.json
$zip="CloneVoiceStudio-Windows-x64-$Backend-Portable.zip"
Mark-Stage "Packaging $Backend portable ZIP"
if ($Backend -eq 'xpu') {
    # XPU includes large pre-compressed native libraries. Fast ZIP reduces CPU-bound
    # compression and skips a second full read/decompression of the entire archive.
    Run-Python scripts/package.py dist/CloneVoiceStudio "dist/$zip" --fast
} else {
    Run-Python scripts/package.py dist/CloneVoiceStudio "dist/$zip"
}
Write-Host "Built dist/$zip. GPU inference not validated by this build."
