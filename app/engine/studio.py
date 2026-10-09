"""Overlap-only SepFormer separation with ECAPA permutation matching and manual review."""
import itertools, json
from pathlib import Path
import numpy as np
from app.config import atomic_json, digest
from app.audio_io.files import read_audio, resample, write_audio, align_length
from app.services.models import model_dir, check_group

def regions(turns):
    boundaries=sorted({float(t[k]) for t in turns for k in ('start','end')}); out=[]
    for a,b in zip(boundaries,boundaries[1:]):
        speakers=sorted({t['speaker'] for t in turns if t['start']<b and t['end']>a})
        if speakers:
            if out and out[-1]['speakers']==speakers and abs(out[-1]['end']-a)<1e-6: out[-1]['end']=b
            else: out.append({'start':a,'end':b,'speakers':speakers})
    return out

class Studio:
    def __init__(self, project, source=None):
        self.directory=Path(project); self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'project.json'
        self.data=json.loads(self.path.read_text()) if self.path.exists() else {
            'version':1,'source':str(source),'turns':[],'assignments':{},'tracks':{},'review':[],
            'separation_backend':'cpu','background':'not independently separated','models':{}}
        self.save()
    def save(self): atomic_json(self.path,self.data)
    def diarize(self,progress=lambda m:None):
        check_group('diarization'); import torch
        from pyannote.audio import Pipeline
        wave,sr=read_audio(self.data['source'],16000)
        progress('Community-1: local CPU diarization')
        pipeline=Pipeline.from_pretrained(str(model_dir('pyannote/speaker-diarization-community-1')))
        output=pipeline({'waveform':torch.from_numpy(wave)[None],'sample_rate':sr})
        annotation=getattr(output,'speaker_diarization',output)
        self.data['turns']=[{'start':float(t.start),'end':float(t.end),'speaker':s}
                           for t,_,s in annotation.itertracks(yield_label=True)]
        self.data['source_sha256']=digest(self.data['source']); self.save(); del pipeline
        return self.data
    def split_tracks(self,progress=lambda m:None,check=lambda:None):
        check_group('separation'); import torch
        torch.set_num_threads(4)
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        wave,sr=read_audio(self.data['source'],22050)
        turns=self.data['turns']; spans=regions(turns); speakers=sorted({t['speaker'] for t in turns})
        if not speakers: raise ValueError('Run diarization or add manual turns first')
        for t in turns:
            if not 0<=t['start']<t['end']<=len(wave)/sr+.001: raise ValueError('Invalid timeline region')
        tracks={s:np.zeros_like(wave) for s in speakers}; templates={}; review=[]; source_sha=digest(self.data['source'])
        separator=SepformerSeparation.from_hparams(source=str(model_dir('speechbrain/sepformer-wsj02mix')),
            savedir=str(self.directory/'separator'),run_opts={'device':'cpu'},
            local_strategy=LocalStrategy.NO_LINK)
        embedder=EncoderClassifier.from_hparams(source=str(model_dir('speechbrain/spkrec-ecapa-voxceleb')),
            savedir=str(self.directory/'embedder'),run_opts={'device':'cpu'},
            overrides={'pretrained_path':str(model_dir('speechbrain/spkrec-ecapa-voxceleb'))},local_strategy=LocalStrategy.NO_LINK)
        def embedding(w):
            v=embedder.encode_batch(torch.from_numpy(resample(w,sr,16000))[None]).flatten().detach().cpu().numpy()
            return v/(np.linalg.norm(v)+1e-9)
        with torch.inference_mode():
            for speaker in speakers:
                solo=[x for x in spans if x['speakers']==[speaker] and x['end']-x['start']>=.7]
                if solo:
                    best=max(solo,key=lambda x:x['end']-x['start'])
                    templates[speaker]=embedding(wave[int(best['start']*sr):int(min(best['end'],best['start']+8)*sr)])
            for index,span in enumerate(spans):
                check(); progress(f'Track region {index+1}/{len(spans)}')
                a=int(span['start']*sr); b=min(len(wave),int(span['end']*sr)); ids=span['speakers']
                if len(ids)==1: tracks[ids[0]][a:b]=wave[a:b]; continue
                if len(ids)!=2 or span['end']-span['start']>20:
                    # Preserve only once, not once per speaker; users review unsupported regions.
                    tracks[ids[0]][a:b]=wave[a:b]; review.append({**span,'reason':'unsupported overlap (2 speakers, <=20s only); original preserved once'}); continue
                left=max(0,a-int(.5*sr)); right=min(len(wave),b+int(.5*sr))
                key=source_sha+f'-{left}-{right}-sepformer-v1'
                cache=self.directory/(key+'.npz')
                try:
                    if cache.exists():
                        with np.load(cache,allow_pickle=False) as z: sources=z['sources']
                    else:
                        est=separator.separate_batch(torch.from_numpy(resample(wave[left:right],sr,8000))[None])
                        if est.shape[-1]!=2: raise RuntimeError('Unexpected number of sources')
                        sources=np.stack([align_length(resample(est[0,:,i].detach().cpu().numpy(),8000,sr),right-left)[0] for i in range(2)])
                        np.savez(cache,sources=sources)
                    vectors=[embedding(s) for s in sources]
                    if all(s in templates for s in ids):
                        candidates=[(sum(float(vectors[i]@templates[ids[p[i]]]) for i in range(2)),p) for p in itertools.permutations(range(2))]
                        candidates.sort(reverse=True); score,perm=candidates[0]; margin=score-candidates[1][0]
                        if margin<.2 or score/2<.3: raise RuntimeError(f'Uncertain source assignment: cosine margin {margin:.3f}; manual review required')
                    else: raise RuntimeError('No clean solo speech for speaker matching; manual review required')
                    for i,who in enumerate(perm):
                        core=sources[i,a-left:b-left].copy(); n=min(int(.01*sr),len(core)//2)
                        if n: core[:n]*=np.linspace(0,1,n); core[-n:]*=np.linspace(1,0,n)
                        tracks[ids[who]][a:b]=core
                except Exception as exc:
                    tracks[ids[0]][a:b]=wave[a:b]; review.append({**span,'reason':str(exc),'source_cache':str(cache) if cache.exists() else None,
                        'crop_start':left,'policy':'original kept once; do not convert without review'})
        for speaker,track in tracks.items():
            path=self.directory/(speaker+'.wav'); write_audio(path,track,sr); self.data['tracks'][speaker]=str(path)
        self.data['review']=review; self.save(); return self.data
    def resolve_overlap(self,index,swap=False,keep_original=False):
        review=self.data['review'][index]
        if keep_original:
            review['resolved']='keep_original'; self.save(); return
        if not review.get('source_cache'): raise ValueError('No separated sources available; preserve original or reprocess.')
        wave,sr=read_audio(self.data['source'],22050)
        with np.load(review['source_cache'],allow_pickle=False) as z: sources=z['sources']
        ids=list(review['speakers'])
        if len(ids)!=2 or sources.shape[0]!=2: raise ValueError('Only two-source manual matching is supported')
        if swap: ids.reverse()
        a=int(review['start']*sr); b=min(len(wave),int(review['end']*sr)); left=review['crop_start']
        for i,speaker in enumerate(ids):
            path=self.data['tracks'][speaker]; track,_=read_audio(path,sr)
            track[a:b]=sources[i,a-left:b-left]; write_audio(path,track,sr)
        self.data['review'].pop(index); self.save()
    def preview_source(self,index,source):
        review=self.data['review'][index]
        if not review.get('source_cache'): raise ValueError('No separated source cache')
        with np.load(review['source_cache'],allow_pickle=False) as z: wave=z['sources'][source]
        path=self.directory/f'review_{index}_source_{source}.wav'; write_audio(path,wave,22050); return path
    def render(self,engine,profiles,output,steps=20,progress=lambda m:None,check=lambda:None):
        if any(r.get('resolved')!='keep_original' for r in self.data['review']): raise ValueError('Resolve or remove uncertain overlap regions before voice conversion. Original tracks are available for review.')
        self.data['models']['conversion']={'variant':engine.model_kind,'backend':engine.device,'precision':engine.precision,'steps':steps} if hasattr(engine,'model_kind') else {'variant':'test double'}
        wave,sr=read_audio(self.data['source'],engine.sample_rate); mixed=np.zeros_like(wave)
        for speaker,path in self.data['tracks'].items():
            check(); settings=self.data['assignments'].get(speaker,{})
            if settings.get('mute'): continue
            track,_=read_audio(path,sr); ident=settings.get('profile')
            if ident:
                profile=profiles.get(ident); output_track=np.zeros_like(track)
                for span in regions(self.data['turns']):
                    if speaker not in span['speakers']: continue
                    a=int(span['start']*sr); b=min(len(track),int(span['end']*sr))
                    for start in range(a,b,sr*12):
                        check(); end=min(b,start+sr*12); progress(f'Converting {speaker} {start/sr:.1f}s')
                        converted=engine.convert(track[start:end],profile,steps)
                        converted,_=align_length(converted,end-start); output_track[start:end]=converted
                track=output_track
            mixed[:min(len(track),len(mixed))]+=track[:len(mixed)]*float(settings.get('gain',1))
        # Preserve nonspeech regions and explicitly reviewed original overlaps.
        occupied=np.zeros(len(wave),dtype=bool)
        for span in regions(self.data['turns']): occupied[int(span['start']*sr):int(span['end']*sr)]=True
        mixed[~occupied]=wave[~occupied]
        for review in self.data['review']:
            a=int(review['start']*sr); b=min(len(wave),int(review['end']*sr)); mixed[a:b]=wave[a:b]
        write_audio(output,mixed,sr); self.data['output']=str(output); self.save(); return self.data
