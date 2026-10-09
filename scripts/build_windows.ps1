param([ValidateSet('cpu','xpu')][string]$Backend='xpu',[switch]$SkipInstall)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Run-Python { & python @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $args" } }
function Mark-Stage([string]$Message) { Write-Host ("[{0}] {1}" -f (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), $Message) }
function Test-Frozen([string]$Argument, [int]$TimeoutSeconds=240, [switch]$AllowTimeout) {
    $exe = 'dist/CloneVoiceStudio/CloneVoiceStudio.exe'
    Mark-Stage "Launching $Argument (timeout: $TimeoutSeconds seconds)"
    $process = Start-Process $exe -ArgumentList $Argument -PassThru
    $exited = $process.WaitForExit($TimeoutSeconds * 1000)
    if (-not $exited) {
        try { $process.Kill($true) } catch {}
        $message = "Frozen $Argument timed out after $TimeoutSeconds seconds (no application log implies a pre-Python startup stall)."
        if ($AllowTimeout) { Write-Warning $message; return $false }
        throw $message
    }
    $log = 'validation-data/logs/app.log'
    if (Test-Path $log) {
        Mark-Stage 'Frozen startup diagnostic log:'
        Get-Content $log -Tail 30 | ForEach-Object { Write-Host $_ }
    }
    if ($process.ExitCode -ne 0) { throw "Frozen $Argument failed: $($process.ExitCode)" }
    return $true
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
$env:CLONEVOICE_DATA=(Join-Path (Get-Location) 'validation-data')
# Ordinary hosted Windows runners have no Intel Arc GPU. A PyInstaller XPU
# executable can stall before Python logging on such a runner (observed in run
# 37955968546), even though the XPU wheel and package build succeeded.
# Preserve a downloadable *build-only* artifact but never call it XPU-validated.
# Actual GPU inference still requires a physical Intel XPU smoke/benchmark.
$validation = [ordered]@{
    backend = $Backend
    source_smoke = 'passed'
    frozen_gui = 'not_run'
    frozen_imports = 'not_run'
    xpu_device_present_on_runner = $null
    real_xpu_inference = 'NOT TESTED'
}
$allowRunnerTimeout = $false
if ($Backend -eq 'xpu') {
    Mark-Stage 'Checking that the installed PyTorch wheel and packaged binaries contain XPU support'
    Run-Python -c "import torch; assert torch.version.xpu is not None, 'Not an Intel XPU PyTorch wheel'; print('XPU wheel:', torch.__version__)"
    $xpuFiles = @(Get-ChildItem 'dist/CloneVoiceStudio/_internal/torch/lib' -Filter '*xpu*.dll' -ErrorAction SilentlyContinue)
    if ($xpuFiles.Count -lt 1) { throw 'XPU native DLLs missing from portable package.' }
    Mark-Stage "Found $($xpuFiles.Count) packaged XPU native DLL(s)"
    $xpuProbe = @(& python -c "import torch; print('yes' if torch.xpu.is_available() else 'no')")
    if ($LASTEXITCODE -ne 0) { throw 'Unable to check Intel XPU hardware availability' }
    $hasXpu = $xpuProbe -contains 'yes'
    $validation.xpu_device_present_on_runner = $hasXpu
    $allowRunnerTimeout = ($env:GITHUB_ACTIONS -eq 'true') -and (-not $hasXpu)
    Mark-Stage 'Checking XPU source-model imports'
    Run-Python launcher.py --engine-import-test
    $validation.source_model_imports = 'passed'
}
if ($allowRunnerTimeout) {
    Mark-Stage 'Attempting frozen XPU startup on GPU-less CI (90s best-effort; not a hardware validation)'
    $guiPass = Test-Frozen '--smoke-test' 90 -AllowTimeout
    $validation.frozen_gui = if ($guiPass) { 'passed_on_gpu_less_runner' } else { 'TIMEOUT_UNVERIFIED' }
    if ($guiPass) {
        $importsPass = Test-Frozen '--engine-import-test' 90 -AllowTimeout
        $validation.frozen_imports = if ($importsPass) { 'passed_on_gpu_less_runner' } else { 'TIMEOUT_UNVERIFIED' }
    } else {
        $validation.frozen_imports = 'SKIPPED_AFTER_STARTUP_TIMEOUT'
    }
    if (-not $guiPass -or ($validation.frozen_imports -ne 'passed_on_gpu_less_runner')) {
        Write-Warning 'XPU portable is BUILD-ONLY / UNVERIFIED on hosted CI; run --diagnose and a real model conversion on Intel hardware.'
    }
} else {
    Mark-Stage 'Testing packaged GUI without background Intel device diagnostics'
    $null = Test-Frozen '--smoke-test'
    $validation.frozen_gui = 'passed'
    Remove-Item Env:QT_QPA_PLATFORM
    Mark-Stage 'Testing packaged model imports'
    $null = Test-Frozen '--engine-import-test'
    $validation.frozen_imports = 'passed'
}
if (Test-Path Env:QT_QPA_PLATFORM) { Remove-Item Env:QT_QPA_PLATFORM }
$validation | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 'dist/CloneVoiceStudio/build-verification.json'
New-Item -ItemType Directory -Force validation | Out-Null
$validation | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 'validation/build-verification.json'
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
