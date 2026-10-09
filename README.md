# CloneVoice Studio — 0.1.0 preview

Windows desktop application for local, zero-shot voice conversion. **Preview, not a completed implementation of the full specification.** Real Seed-VC model integration; no pitch-only or passthrough substitute presented as cloning.

## Windows portable downloads

[Actions → Windows Portable](https://github.com/gaong5247-cmd/Voicecloning/actions/workflows/windows.yml). Download the CPU or XPU artifact after its build succeeds, unzip, and run `CloneVoiceStudio.exe`. Python is bundled. Model weights are not bundled. Intel graphics drivers are still required. CI GUI startup does not prove voice conversion or XPU operation.

1. Model Manager → `voice` → review license confirmation → Download.
2. Voice Profiles → clean reference recording, name, usage permission confirmation → Create.
3. Advanced Inference → source file, profile, backend and output WAV → Convert.
4. Playback uses the Windows default media player. Video remux preserves the encoded video stream and replaces the audio with AAC.

Data is stored under `%LOCALAPPDATA%\CloneVoiceStudio`. Set `CLONEVOICE_DATA` to a writable folder for portable data. References, features, sessions, logs, and projects remain local. Download credentials are not saved. There is no audio upload or cloud inference.

## Source run / build

Python 3.11 recommended. From PowerShell in the repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/xpu.txt
python -m pip install -r requirements/base.txt -r requirements/studio.txt
python launcher.py
# CPU wheel alternative: requirements/cpu.txt
# Build with Python 3.11 and PowerShell:
./scripts/build_windows.ps1 -Backend xpu
```

CLI: `python launcher.py --diagnose`; `--download voice`; create a profile with `--reference reference.wav --profile MyVoice --consent`, then use its returned ID with `--source input.wav --profile ID --output output.wav --backend cpu --steps 20`.

## What this preview provides

- PySide6 desktop UI, bilingual navigation and bilingual controls; dark/light theme.
- WAV/MP3/FLAC/OGG/M4A/MP4/MKV decoding (FFmpeg for compressed/video fallback), WAV output and video remux.
- Seed-VC V1 22.05kHz zero-shot conversion, bounded chunks, reference feature disk cache, profiles and explicit permission check.
- CPU / XPU device selection, FP32 / FP16 / BF16, model load and inference timing, safe precision / CPU retries.
- Download manager: immutable upstream model revisions and SHA256 verification of large weight files.
- Persistent cancellable inference sessions; validated completed chunk reuse on resume.
- Experimental buffered microphone → worker → output stream with bounded rings, overload counters, noise gate, mute, original monitor and software latency estimate. This uses the **file V1 model**, not the optimized tiny streaming engine; hardware latency is unmeasured.
- Community-1 local CPU diarization adapter, editable turn table and colored overlap timeline.
- SepFormer 2-source CPU separation on overlapping regions only, ECAPA source permutation matching, cached results, original preservation on uncertain matching. These model integrations require additional validation.
- Speaker-profile assignment, track gain, timeline reconstruction and mixing. Review blocks rendering when source identity is uncertain.

## Limitations / unsupported

No claimed Windows/XPU speed, Korean separation accuracy, DER, SI-SDRi, CER or speaker similarity unless measured in `docs/VALIDATION.md`. CPU throughput varies and realtime may be slower than input. Buffers are bounded: overload is reported as drops; lossless indefinite input is impossible when processing is slower than capture.

3-speaker separation, OpenVINO inference, NPU, INT8, F0/pitch/formants, neural noise reduction, full stereo/background reconstruction, DAW-style drag editing/solo, end-to-end hardware latency and automatic chunk tuning are **not implemented**. Community-1 gated model downloads need an approved Hugging Face token; accept terms on its model page first. Clean single-speaker reference selection is the user's responsibility; energy screening is not neural VAD or speaker verification.

See [model research](docs/MODELS.md), [validation status](docs/VALIDATION.md), and [third-party notices](THIRD_PARTY_NOTICES.md). The redistributed Seed-VC code is GPL-3.0; this project uses GPL-3.0. FFmpeg's Windows essentials package is a GPL build; accompanying source links and license obligations apply.
