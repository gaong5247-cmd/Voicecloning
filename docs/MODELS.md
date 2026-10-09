# Upstream model audit (2026-10-09)

| Component | Evidence / status | Integration and limitations |
|---|---|---|
| Seed-VC | GPL-3.0 code and model, original archived 2025-11-21, code 51383ef (2025-04-20) | V1 diffusion transformer, Whisper-small content, CAMPPlus 192-dimensional style, BigVGAN 22.05kHz vocoder. Reference 1–30s upstream; application restricts usable reference to 1–25s. PyTorch upstream 2.4.0, Python 3.10; this adapter targets tested 2.8.0 / Python 3.11 packaging. Windows/XPU compatibility needs model/hardware tests. |
| Seed-VC Realtime fork | GPL-3.0, 58d0ca3 (2026-07-16), active changes after original archived | Model engine separated from GUI, ring-buffer workers, CUDA recommended. Not used as XPU validation evidence. Its README describes ~690ms measured on author's setup and Windows timestamp caveats; these are not this app's measurements. |
| SpeechBrain | Apache-2.0, repository not archived; runtime 1.0.3 pinned | SepFormer WSJ0-2Mix has exactly 2 outputs, mono 8kHz input. English read-speech dataset; Korean / environmental / 3-speaker capability cannot be inferred. CPU backend initially. ECAPA VoxCeleb embedding used for source assignment, threshold scores are heuristics, not calibrated probabilities. |
| pyannote.audio | MIT code, repository not archived; runtime 4.0.1 pinned | Community-1 CC-BY-4.0 gated-access model, mono 16kHz. Non-exclusive annotation used to retain overlapping turns. Local CPU pipeline; no direct XPU support claim. Requires user to accept model terms and provide token only at explicit download. |
| F0 candidates | Seed V1 44k supports RMVPE upstream; CREPE adds another engine | Current non-F0 22k model does not use either. Pitch/F0 controls disabled; pretending they change the model would be incorrect. |
| OpenVINO/NPU/INT8 | Device discovery optional only | No validated exported graphs, calibration audio or quality comparison. Not selectable for conversion. |

Weight revisions, licenses, and large-file SHA256 values were obtained from each publisher's Hugging Face model API and pinned in `config/models.lock.json`. Small Git blobs currently rely on immutable revision URLs; no independent SHA256 is claimed for those files.

Input/output: float32 mono waveform in engine adapter; 22,050 Hz output; context-limited chunks including reference conditioning. Full-file outputs receive explicit length alignment, with raw duration differences recorded. Reference conditioning arrays are persisted as non-pickle NPZ keyed by model revision and reference hash. No model training is performed. Memory/speed must be measured on target hardware; current measured values are in VALIDATION.md.

Primary sources:
- https://github.com/Plachtaa/seed-vc
- https://github.com/jiaheguo521/seed-vc-realtime
- https://github.com/speechbrain/speechbrain
- https://github.com/pyannote/pyannote-audio
- https://huggingface.co/Plachta/Seed-VC
- https://huggingface.co/speechbrain/sepformer-wsj02mix
- https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
- https://docs.pytorch.org/tutorials/unstable/inductor_windows.html
