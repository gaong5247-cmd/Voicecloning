from pathlib import Path
import os, sys, json, hashlib, tempfile

VERSION = '0.1.0-preview'
ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get('CLONEVOICE_DATA', Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'CloneVoiceStudio'))

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def configure_offline():
    for k, v in {'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
                 'PYANNOTE_METRICS_ENABLED':'0','DO_NOT_TRACK':'1',
                 'HF_HOME':str(DATA / 'models' / 'hub')}.items(): os.environ[k] = v
