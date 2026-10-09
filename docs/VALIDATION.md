# Validation — 2026-10-09

## Executed

- 23 infrastructure unit/integration tests passed on Linux; latest run recorded separately. Initial 21-test run: 1.62 seconds.
- Same 21 tests passed on Windows CPU GitHub runner (1.18 seconds).
- PySide6 GUI initialization and clean smoke exit passed on Linux and Windows CPU source environment.
- Seed-VC voice model download completed; large files verified against publisher SHA256 values.
- Real Seed-VC V1 conversion completed on Linux CPU: FFmpeg flite synthetic English input/reference (permission-safe fixtures), 4.38254 seconds input; output WAV written at 22,050 Hz, non-finite outputs rejected, original input untouched.
- First full file operation: 52.20984 seconds, model loading 28.60080 seconds, conversion 21.15054 seconds, model RTF 4.82609, total RTF 11.91315. 6 diffusion steps, CPU FP32, 4 Torch threads. These are this Linux shared runner's measurements, not an Intel notebook benchmark.
- Raw output duration difference -0.00558 seconds, corrected to original sample count.
- Further repeat conversion with persisted reference cache: cache hits 1 then 2, model conversion 22.28652 then 26.76886 seconds. RTF 5.08530 then 6.10807. Shared-runner contention affects these results. See cpu-benchmark.json for full measurements.
- SpeechBrain and pyannote.audio runtime imports passed. Real SepFormer 2-second CPU inference returned [1,16000,2] finite samples in 2.56283s. ECAPA returned [1,1,192] finite embeddings. A 6-second synthetic English two-speaker fixture with manual ground-truth turns completed overlap-only separation, source matching and two-track reconstruction in 9.02412s. Best-permutation mean SI-SDR improvement was 8.85616dB on that single fixture only. No automatic diarization or Korean accuracy conclusion follows. Full scores are in separation-benchmark.json.

## Build status

Initial GitHub Actions run 37947640028: Windows CPU dependency installation, tests and GUI source smoke passed; packaging blocked by FFmpeg download URL 404. URL corrected to the verified GitHub release asset; next run pending. No completed EXE artifact is claimed yet.

## Not verified / not supported

Windows physical audio devices, frozen-model inference, virtual-cable routing, Intel Arc XPU model execution / FP16 / BF16 / OOM behavior, latency, long-duration memory stability, Korean/English/Japanese linguistic quality, diarization DER, separation SI-SDRi, WER/CER and speaker similarity require additional testing. Linux source conversion does not establish Windows frozen model compatibility.

OpenVINO / NPU / INT8 / 3-speaker separation are not implemented. No <100ms latency claim. No production-release or full-spec-completion claim.
