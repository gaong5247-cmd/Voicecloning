# Third-party notices

CloneVoice Studio is GPL-3.0, see LICENSE. Modified Seed-VC code is included under vendor/seed_vc. Original: https://github.com/Plachtaa/seed-vc, commit 51383efd921027683c89e5348211d93ff12ac2a8. Modifications: offline/model-revision-controlled loaders; FP32 semantic encoder; explicit backend; lazy DAC import; safer checkpoint loading / mismatch rejection. Original nested component copyright notices and license files are preserved. Seed model weights: GPL-3.0 (https://huggingface.co/Plachta/Seed-VC).

Seed-VC Realtime (https://github.com/jiaheguo521/seed-vc-realtime, GPL-3.0), commit 58d0ca372cc76f60a798d9f84611f053f9029f3a, was studied but not incorporated. It is CUDA oriented; no proven XPU realtime claim follows from it.

Downloaded separately: Whisper small (OpenAI, Apache-2.0); CAMPPlus (FunASR, Apache-2.0); BigVGAN V2 (NVIDIA, MIT); SpeechBrain SepFormer and ECAPA (Apache-2.0); pyannote Community-1 (CC-BY-4.0 with gated access terms). Exact weight revisions and available large-file SHA256 values are in config/models.lock.json. No gated model weights are redistributed.

Runtime dependencies: PyTorch/torchaudio BSD-style; NumPy/SciPy BSD; PySide6 LGPL-3.0/GPL/commercial (dynamically loaded unmodified libraries); librosa ISC; sounddevice MIT; SoundFile BSD-3-Clause and libsndfile LGPL; Transformers Apache-2.0; Hugging Face Hub Apache-2.0; SpeechBrain Apache-2.0; pyannote.audio MIT; PyInstaller GPL with bootloader exception. Dependency distributions carry their respective notices; users may replace the PySide shared libraries in the one-directory build.

FFmpeg Windows binary: https://www.gyan.dev/ffmpeg/builds/ (7.1.1 essentials GPL build). FFmpeg project/source: https://ffmpeg.org/ and https://github.com/FFmpeg/FFmpeg/tree/n7.1.1 ; packaging/source scripts: https://github.com/GyanD/codexffmpeg. A redistributor must provide corresponding source and preserve applicable notices. Portable source code accompanies the package; exact application source also remains available at https://github.com/gaong5247-cmd/Voicecloning .

Tiny model additionally uses Facebook XLS-R-300M (Apache-2.0), truncated to 12 encoder layers, and the HiFT checkpoint published in Seed-VC (original CosyVoice component). Synthetic test audio uses Flite (Carnegie Mellon University; permissive BSD-style license), https://github.com/festvox/flite .
