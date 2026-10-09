import time
from pathlib import Path
import numpy as np
import soundfile as sf
from app.audio_io.files import read_audio, write_audio, align_length
from app.config import digest, atomic_json

PRESETS={'FAST':{'steps':6,'chunk_seconds':10},'BALANCED':{'steps':20,'chunk_seconds':12},
         'QUALITY':{'steps':40,'chunk_seconds':18}}

def convert_file(engine, source, profile, output, job, progress=lambda fraction,message:None):
    started=time.perf_counter(); settings=job.record['settings']
    source=Path(source); output=Path(output)
    if source.resolve()==output.resolve(): raise ValueError('Output cannot overwrite the input')
    wave,sr=read_audio(source,engine.sample_rate)
    source_hash=digest(source); ref_hash=digest(profile['reference'])
    identity={'source':source_hash,'reference':ref_hash,'model':profile['model'],'settings':settings}
    if job.record.get('identity') not in (None,identity): raise ValueError('Source, model or settings changed. Create a new session.')
    job.record['identity']=identity; job.record['output']=str(output); job.state('running')
    chunk=int(sr*settings.get('chunk_seconds',12)); pad=int(sr*.2)
    if not 1<=chunk/sr<=18: raise ValueError('Chunk duration must be 1–18 seconds')
    result=np.zeros_like(wave); errors=[]
    try:
        for index,start in enumerate(range(0,len(wave),chunk)):
            job.check(); end=min(len(wave),start+chunk)
            a=max(0,start-pad); b=min(len(wave),end+pad); path=job.path/f'chunk_{index:06d}.wav'
            saved=job.record['chunks'].get(str(index))
            if saved and path.exists() and digest(path)==saved['sha256']:
                converted,_=sf.read(path,dtype='float32'); difference=saved['length_difference']
            else:
                progress(start/len(wave),f'Converting chunk {index+1}')
                converted=engine.convert(wave[a:b],profile,settings.get('steps',20))
                aligned,difference=align_length(converted,b-a); converted=aligned[start-a:end-a]
                write_audio(path,converted,sr)
                job.record['chunks'][str(index)]={'sha256':digest(path),'length_difference':difference}
                job.save()
            if len(converted)!=end-start: raise RuntimeError('Invalid cached chunk length')
            result[start:end]=converted; errors.append(difference/sr)
        job.check()
        # Fade only outer boundaries. Internal chunks include model context; no overlapping speech duplicated.
        n=min(int(.005*sr),len(result)//2)
        if n: result[:n]*=np.linspace(0,1,n); result[-n:]*=np.linspace(1,0,n)
        write_audio(output,result*settings.get('gain',1),sr)
        seconds=time.perf_counter()-started
        job.record['metrics']={'total_seconds':seconds,'audio_seconds':len(wave)/sr,'rtf':seconds/(len(wave)/sr),
            'chunk_length_differences_seconds':errors,'engine':engine.stats,
            'hardware_latency_ms':None,'quality_metrics':'not measured'}
        job.state('complete'); progress(1,'Saved '+str(output)); return job.record
    except Exception as exc:
        job.record['errors'].append({'type':type(exc).__name__,'message':str(exc)})
        job.state('cancelled' if job.cancelled.is_set() else 'failed'); raise
