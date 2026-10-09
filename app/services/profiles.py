import uuid, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from app.config import DATA, atomic_json, digest
from app.audio_io.files import read_audio, write_audio

MODEL_VERSION = 'seed-v1-51383ef-22k'

class Profiles:
    def __init__(self, root=None):
        self.root = Path(root or DATA / 'VoiceProfiles'); self.root.mkdir(parents=True, exist_ok=True)
    def list(self):
        result = []
        for path in self.root.glob('*/manifest.json'):
            try: result.append(json.loads(path.read_text(encoding='utf-8')))
            except (ValueError, OSError): continue
        return result
    def create(self, name, reference, consent=False):
        if not consent: raise ValueError('Confirm you own this voice or have permission to use it.')
        if not name.strip(): raise ValueError('Profile name is required')
        audio, sr = read_audio(reference)
        # Energy activity screening, explicitly not a neural VAD / speaker detector.
        frame = int(sr*.02); spans = []
        for i in range(0,len(audio),frame):
            if np.sqrt(np.mean(audio[i:i+frame]**2)) > .008: spans.append((max(0,i-frame*5),min(len(audio),i+frame*6)))
        if not spans: raise ValueError('Reference is silent or too quiet')
        active = np.zeros(len(audio),dtype=bool)
        for a,b in spans: active[a:b] = True
        cleaned = audio[active][:sr*25]
        if len(cleaned)<sr: raise ValueError('Reference needs at least 1 second of usable audio; 5–25 seconds recommended.')
        ident = uuid.uuid4().hex; directory = self.root/ident; directory.mkdir()
        path = directory/'reference.wav'; write_audio(path, cleaned, sr)
        quality = {'duration':len(cleaned)/sr,'original_duration':len(audio)/sr,
            'rms_db':float(20*np.log10(np.sqrt(np.mean(cleaned**2))+1e-9)),
            'clipped_fraction':float(np.mean(np.abs(audio)>=.999)),
            'activity_method':'energy screening; not speaker verification',
            'multiple_speakers':'unchecked; use a clean single speaker reference'}
        data = {'id':ident,'name':name.strip(),'version':1,'model':MODEL_VERSION,
            'created':datetime.now(timezone.utc).isoformat(),'reference':str(path),
            'reference_sha256':digest(path),'quality':quality,'consent':True,'settings':{}}
        atomic_json(directory/'manifest.json',data); return data
    def get(self, ident):
        if not any(x['id']==ident for x in self.list()): raise ValueError('Unknown profile')
        path = self.root/ident/'manifest.json'; data=json.loads(path.read_text(encoding='utf-8'))
        if digest(data['reference']) != data['reference_sha256']:
            data['reference_sha256']=digest(data['reference']); atomic_json(path,data)
            for cache in (self.root/ident/'features').glob('*.npz'): cache.unlink()
        return data
