from pathlib import Path
import math, shutil, subprocess, os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from app.config import ROOT

FORMATS = {'.wav','.mp3','.flac','.ogg','.m4a','.mp4','.mkv'}

def ffmpeg():
    bundled = ROOT / 'bin' / ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
    exe = str(bundled) if bundled.exists() else shutil.which('ffmpeg')
    if not exe: raise RuntimeError('FFmpeg is missing. Install the complete portable build or set PATH.')
    return exe

def run_ffmpeg(args):
    result = subprocess.run([ffmpeg(), '-nostdin', '-hide_banner', '-loglevel', 'error', *map(str,args)],
        capture_output=True, creationflags=0x08000000 if os.name == 'nt' else 0)
    if result.returncode: raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-3000:])

def resample(wave, src, dst):
    if src == dst: return np.asarray(wave, dtype=np.float32)
    g = math.gcd(int(src), int(dst))
    return resample_poly(wave, dst//g, src//g, axis=0).astype(np.float32)

def read_audio(path, sr=22050):
    path = Path(path)
    if path.suffix.lower() not in FORMATS: raise ValueError('Unsupported media format')
    if not path.is_file(): raise FileNotFoundError(path)
    try:
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        audio = resample(audio.mean(axis=1), rate, sr)
    except (RuntimeError, sf.LibsndfileError):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            decoded = Path(temp)/'decoded.wav'
            run_ffmpeg(['-y','-i',path,'-vn','-ac','1','-ar',sr,decoded])
            audio, _ = sf.read(decoded, dtype='float32')
    if not len(audio) or not np.isfinite(audio).all(): raise ValueError('Empty or invalid audio')
    return audio, sr

def write_audio(path, audio, sr):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all(): raise ValueError('Non-finite output rejected')
    peak = float(np.max(np.abs(audio), initial=0))
    if peak > .98: audio = audio * (.98/peak)
    import tempfile
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.wav'); os.close(fd)
    try:
        sf.write(name, audio, sr, subtype='PCM_24'); os.replace(name, path)
    finally: Path(name).unlink(missing_ok=True)

def align_length(audio, length):
    if not len(audio): raise ValueError('Model produced empty audio')
    difference = len(audio)-length
    if abs(difference) <= 1024:
        return np.pad(audio, (0,max(0,length-len(audio))))[:length], difference
    import librosa
    corrected = librosa.effects.time_stretch(np.asarray(audio, dtype=np.float32), rate=len(audio)/length)
    return np.pad(corrected, (0,max(0,length-len(corrected))))[:length], difference

def remux(video, audio, output):
    if Path(video).resolve() == Path(output).resolve(): raise ValueError('Cannot overwrite source video')
    run_ffmpeg(['-y','-i',video,'-i',audio,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-shortest',output])
