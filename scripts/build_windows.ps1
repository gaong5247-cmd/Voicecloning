param([ValidateSet('cpu','xpu')][string]$Backend='xpu',[switch]$SkipInstall)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Run-Python { & python @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $args" } }
if (-not $SkipInstall) {
    Run-Python -m pip install -r "requirements/$Backend.txt"
    Run-Python -m pip install -r requirements/base.txt -r requirements/studio.txt -r requirements/build.txt
}
Run-Python -m pip check
Run-Python -m pytest tests -q
$env:QT_QPA_PLATFORM='offscreen'
Run-Python launcher.py --smoke-test
Remove-Item Env:QT_QPA_PLATFORM
New-Item -ItemType Directory -Force bin | Out-Null
if (-not (Test-Path bin/ffmpeg.exe)) {
    Invoke-WebRequest 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-essentials_build.zip' -OutFile ffmpeg.zip
    Expand-Archive ffmpeg.zip -DestinationPath ffmpeg-temp -Force
    Copy-Item (Get-ChildItem ffmpeg-temp -Filter ffmpeg.exe -Recurse | Select-Object -First 1).FullName bin/ffmpeg.exe
    Remove-Item ffmpeg.zip
}
Run-Python -m PyInstaller --noconfirm --clean --onedir --windowed --name CloneVoiceStudio --paths vendor/seed_vc `
    --add-data 'vendor/seed_vc;vendor/seed_vc' --add-data 'config;config' --add-binary 'bin/ffmpeg.exe;bin' `
    --collect-all torch --collect-all torchaudio --collect-all sounddevice --collect-all soundfile `
    --collect-all transformers --collect-all librosa --collect-all speechbrain --collect-all pyannote.audio `
    --collect-all torchcodec --hidden-import seed_loader --hidden-import hf_utils `
    --hidden-import modules.flow_matching --hidden-import modules.length_regulator `
    --hidden-import modules.campplus.DTDNN --hidden-import modules.bigvgan.bigvgan launcher.py
$env:QT_QPA_PLATFORM='offscreen'
$smoke = Start-Process 'dist/CloneVoiceStudio/CloneVoiceStudio.exe' -ArgumentList '--smoke-test' -Wait -PassThru
if ($smoke.ExitCode -ne 0) { throw "Frozen GUI smoke failed: $($smoke.ExitCode)" }
Remove-Item Env:QT_QPA_PLATFORM
Copy-Item README.md,LICENSE,THIRD_PARTY_NOTICES.md dist/CloneVoiceStudio/
Copy-Item -Recurse docs dist/CloneVoiceStudio/docs -Force
Copy-Item -Recurse vendor dist/CloneVoiceStudio/source-vendor -Force
Copy-Item -Recurse app dist/CloneVoiceStudio/source-app -Force
Copy-Item launcher.py dist/CloneVoiceStudio/
Copy-Item config/settings.template.json dist/CloneVoiceStudio/settings.template.json
$zip="CloneVoiceStudio-Windows-x64-$Backend-Portable.zip"
Compress-Archive -Path dist/CloneVoiceStudio -DestinationPath "dist/$zip" -Force
Write-Host "Built dist/$zip. GPU inference not validated by this build."
