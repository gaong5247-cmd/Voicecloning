import json, threading
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from app.config import atomic_json, digest
from app.audio_io.files import read_audio, write_audio, resample, align_length
from app.audio_io.ring import RingBuffer
from app.services.profiles import Profiles
from app.services.jobs import Job, Cancelled
from app.engine.inference import convert_file
from app.engine.studio import regions

@pytest.fixture
def audio(tmp_path):
    path=tmp_path/'input.wav'; sr=22050
    # Synthetic tone is only for infrastructure tests, not a speech-quality fixture.
    x=(.2*np.sin(2*np.pi*220*np.arange(sr*2)/sr)).astype(np.float32)
    sf.write(path,x,sr); return path,x,sr

def test_audio_decode_resample(audio):
    path,x,sr=audio; wave,rate=read_audio(path,16000)
    assert rate==16000 and len(wave)==32000 and np.isfinite(wave).all()

def test_invalid_file(tmp_path):
    path=tmp_path/'bad.wav'; path.write_text('not audio')
    with pytest.raises(Exception): read_audio(path)

def test_unsupported_file(tmp_path):
    with pytest.raises(ValueError): read_audio(tmp_path/'text.txt')

def test_empty_file(tmp_path):
    path=tmp_path/'empty.wav'; sf.write(path,np.zeros(0),22050)
    with pytest.raises(ValueError): read_audio(path)

def test_peak_limit(tmp_path):
    path=tmp_path/'peak.wav'; write_audio(path,np.array([-2.,2.]),22050)
    wave,_=sf.read(path); assert max(abs(wave))<=.981

def test_nonfinite_rejected(tmp_path):
    with pytest.raises(ValueError): write_audio(tmp_path/'bad.wav',np.array([np.nan]),22050)

def test_alignment_padding():
    result,diff=align_length(np.ones(100),120); assert len(result)==120 and diff==-20

def test_atomic_json(tmp_path):
    p=tmp_path/'nested/config.json'; atomic_json(p,{'name':'한글'}); atomic_json(p,{'name':'new'})
    assert json.loads(p.read_text())['name']=='new'; assert len(list(p.parent.iterdir()))==1

def test_profile_requires_consent(audio,tmp_path):
    with pytest.raises(ValueError): Profiles(tmp_path/'profiles').create('test',audio[0])

def test_profile_roundtrip_and_hash_invalidation(audio,tmp_path):
    profiles=Profiles(tmp_path/'profiles'); data=profiles.create('한글',audio[0],True)
    assert profiles.get(data['id'])['consent']; path=Path(data['reference']).parent/'features'/'old.npz'
    path.parent.mkdir(); path.write_bytes(b'cached')
    sf.write(data['reference'],audio[1]*.5,audio[2]); changed=profiles.get(data['id'])
    assert changed['reference_sha256']!=data['reference_sha256']; assert not path.exists()

def test_silent_reference(tmp_path):
    path=tmp_path/'silence.wav'; sf.write(path,np.zeros(22050*2),22050)
    with pytest.raises(ValueError): Profiles(tmp_path/'profiles').create('silent',path,True)

def test_ring_wrap():
    ring=RingBuffer(5); ring.write([1,2,3]); assert ring.read(2).tolist()==[1,2]
    ring.write([4,5,6,7]); assert ring.read(5).tolist()==[3,4,5,6,7]

def test_ring_overflow_reports_loss():
    ring=RingBuffer(4); ring.write([1,2,3]); ring.write([4,5,6]); assert ring.dropped==2
    assert ring.read(4).tolist()==[3,4,5,6]; ring.write(np.arange(10)); assert ring.dropped==8

def test_ring_padding():
    ring=RingBuffer(5); ring.write([2]); assert ring.read(3).tolist()==[2,0,0]

def test_ring_concurrency():
    ring=RingBuffer(100)
    def writer():
        for _ in range(1000): ring.write(np.ones(20))
    thread=threading.Thread(target=writer); thread.start()
    while thread.is_alive(): assert np.isfinite(ring.read(10)).all()
    thread.join(); assert 0<=ring.available()<=100

def test_overlap_regions():
    result=regions([{'start':0,'end':3,'speaker':'A'},{'start':2,'end':5,'speaker':'B'}])
    assert result==[{'start':0.,'end':2.,'speakers':['A']},{'start':2.,'end':3.,'speakers':['A','B']},{'start':3.,'end':5.,'speakers':['B']}]

def test_repeated_speaker_does_not_duplicate():
    assert regions([{'start':0,'end':2,'speaker':'A'},{'start':1,'end':3,'speaker':'A'}])==[{'start':0.,'end':3.,'speakers':['A']}]

class TestEngine:
    __test__=False
    sample_rate=22050
    stats={'test_double':True}
    def __init__(self): self.calls=0
    def convert(self,wave,profile,steps): self.calls+=1; return wave*.5

def test_inference_session_resume_cache(audio,tmp_path):
    path,x,sr=audio; profiles=Profiles(tmp_path/'profiles'); profile=profiles.create('test',path,True)
    job=Job(path,profile['id'],{'chunk_seconds':1,'steps':2},tmp_path/'sessions'); engine=TestEngine()
    output=tmp_path/'result.wav'; convert_file(engine,path,profile,output,job)
    calls=engine.calls; restored=Job.resume(job.path/'session.json'); convert_file(engine,path,profile,output,restored)
    assert engine.calls==calls; wave,rate=sf.read(output); assert rate==sr and len(wave)==len(x)
    assert restored.record['state']=='complete'

def test_cancel_keeps_session(audio,tmp_path):
    path,_,_=audio; profiles=Profiles(tmp_path/'profiles'); profile=profiles.create('test',path,True)
    job=Job(path,profile['id'],{'chunk_seconds':1},tmp_path/'sessions'); job.cancelled.set()
    with pytest.raises(Cancelled): convert_file(TestEngine(),path,profile,tmp_path/'result.wav',job)
    assert job.record['state']=='cancelled'

def test_source_immutable(audio,tmp_path):
    path,_,_=audio; profiles=Profiles(tmp_path/'profiles'); profile=profiles.create('test',path,True)
    job=Job(path,profile['id'],{},tmp_path/'sessions'); before=digest(path)
    with pytest.raises(ValueError): convert_file(TestEngine(),path,profile,path,job)
    assert digest(path)==before

def test_model_weight_corruption(tmp_path,monkeypatch):
    import app.services.models as models
    root=tmp_path/'root'; root.mkdir(); (root/'config').mkdir()
    atomic_json(root/'config/models.lock.json',{'repo':{'files':{'model.ckpt':'0'*64},'revision':'a'*40}})
    monkeypatch.setattr(models,'ROOT',root); monkeypatch.setitem(models.GROUPS,'test',['repo'])
    monkeypatch.setattr(models,'model_dir',lambda repo:tmp_path/'models')
    monkeypatch.setattr(models,'selected',lambda repo,file,group=None:True)
    (tmp_path/'models').mkdir(); (tmp_path/'models/model.ckpt').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError,match='integrity'): models.check_group('test')

def test_studio_manual_source_remap(tmp_path):
    from app.engine.studio import Studio
    sr=22050; source=tmp_path/'source.wav'; sf.write(source,np.ones(sr*2)*.1,sr)
    studio=Studio(tmp_path/'project',source); tracks={}
    for speaker in ('A','B'):
        p=tmp_path/(speaker+'.wav'); sf.write(p,np.zeros(sr*2),sr); tracks[speaker]=str(p)
    cache=tmp_path/'sources.npz'; np.savez(cache,sources=np.stack([np.ones(sr)*.2,np.ones(sr)*.4]))
    studio.data['tracks']=tracks; studio.data['review']=[{'start':.5,'end':1.5,'speakers':['A','B'],'crop_start':int(sr*.5),'source_cache':str(cache)}]
    studio.resolve_overlap(0,swap=True)
    wave,_=sf.read(tracks['A']); assert abs(wave[sr]-.4)<.001; assert not studio.data['review']

def test_studio_original_preservation(tmp_path):
    from app.engine.studio import Studio
    sr=22050; source=tmp_path/'source.wav'; x=np.ones(sr*2)*.1; sf.write(source,x,sr)
    studio=Studio(tmp_path/'project',source); track=tmp_path/'track.wav'; y=np.zeros(sr*2);y[:sr]=.2;sf.write(track,y,sr)
    studio.data['turns']=[{'start':0,'end':1,'speaker':'A'}];studio.data['tracks']={'A':str(track)}
    studio.data['review']=[{'start':.5,'end':1,'speakers':['A','B']}];studio.resolve_overlap(0,keep_original=True)
    output=tmp_path/'mix.wav';studio.render(TestEngine(),Profiles(tmp_path/'profiles'),output)
    mixed,_=sf.read(output);assert abs(mixed[100]-.2)<.001;assert abs(mixed[sr]-.1)<.001;assert abs(mixed[int(sr*.75)]-.1)<.001
