"""Zero-shot Seed-VC V1 adapter. Model outputs only: no DSP stand-in for cloning."""
import sys, contextlib, threading, json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from app.config import ROOT, digest
from app.audio_io.files import resample
from app.services.models import check_group
from app.services.profiles import MODEL_VERSION
from app.services.jobs import Cancelled
from app.optimization.devices import select_device, Timer

class SeedEngine:
    sample_rate=22050
    supports_cancellation=True
    capabilities={'pitch':False,'formant':False,'f0':False,'batch':False,'steps':True,
                  'precision':True,'openvino':False,'npu':False}
    def __init__(self, backend='auto', precision='fp32', compile_model=False, model_kind='quality'):
        if model_kind not in ('quality','tiny'): raise ValueError('Unknown model variant')
        self.model_kind=model_kind
        self.device=select_device(backend)
        self.precision=precision; self.compile_model=compile_model
        if precision not in ('fp32','fp16','bf16'): raise ValueError('Unknown precision')
        self.loaded=False; self.lock=threading.RLock(); self.stats={'cache_hits':0,'fallbacks':[]}
        self.refs={}
        if self.device=='cpu' and self.precision!='fp32':
            # CPU FP16 autocast can be extremely slow (and unsupported by some kernels).
            self.stats['fallbacks'].append(f'CPU {self.precision} requested; using FP32 for CPU inference')
            self.precision='fp32'
    def load(self):
        if self.loaded: return
        check_group('realtime' if self.model_kind=='tiny' else 'voice')
        vendor=ROOT/'vendor/seed_vc'
        if str(vendor) not in sys.path: sys.path.insert(0,str(vendor))
        import torch, seed_loader
        import os
        torch.set_num_threads(max(1,min(8,int(os.environ.get("CLONEVOICE_CPU_THREADS","4")))))
        seed_loader.device=torch.device(self.device)
        args=SimpleNamespace(fp16=False,f0_condition=False,checkpoint=None,config=None)
        if self.model_kind=='tiny':
            from app.services.models import model_dir
            args.checkpoint=str(model_dir('Plachta/Seed-VC')/'DiT_uvit_tat_xlsr_ema.pth')
            args.config=str(model_dir('Plachta/Seed-VC')/'config_dit_mel_seed_uvit_xlsr_tiny.yml')
        with Timer(self.device) as timer, torch.inference_mode():
            (self.model,self.semantic,self.f0,self.vocoder,self.campplus,self.mel,self.mel_args)=seed_loader.load_models(args)
        self.stats['load_seconds']=timer.seconds
        if self.compile_model:
            try: self.model.cfm.estimator=torch.compile(self.model.cfm.estimator, dynamic=False)
            except Exception as exc: self.stats['fallbacks'].append('compile disabled: '+str(exc))
        self.loaded=True
    def unload(self):
        self.refs.clear()
        for key in ('model','semantic','f0','vocoder','campplus','mel'):
            if hasattr(self,key): delattr(self,key)
        self.loaded=False
        import gc, torch
        gc.collect()
        if self.device=='xpu': torch.xpu.empty_cache()
    def warm_profile(self,profile):
        import torch
        with self.lock, torch.inference_mode():
            self.load(); self.reference(profile)
    def reference(self, profile):
        import torch, torchaudio
        from app.audio_io.files import read_audio
        sha=digest(profile['reference'])
        key=MODEL_VERSION+'-'+self.model_kind+'-'+sha
        if key in self.refs: self.stats['cache_hits']+=1; return self.refs[key]
        path=Path(profile['reference']).parent/'features'/f'{key}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                result=tuple(torch.from_numpy(z[k]).to(self.device) for k in ('prompt','mel','style'))
            self.stats['cache_hits']+=1
        else:
            wave,_=read_audio(profile['reference'],self.sample_rate)
            ref=torch.from_numpy(wave[:self.sample_rate*25]).unsqueeze(0).to(self.device)
            w16=torch.from_numpy(resample(wave[:self.sample_rate*25],self.sample_rate,16000)).unsqueeze(0)
            sem=self.semantic(w16.to(self.device)); mel=self.mel(ref.float())
            # Kaldi fbank stays on CPU: unsupported XPU FFT paths do not block the neural encoder.
            fb=torchaudio.compliance.kaldi.fbank(w16,num_mel_bins=80,dither=0,sample_frequency=16000)
            fb=fb-fb.mean(dim=0,keepdim=True); style=self.campplus(fb.unsqueeze(0).to(self.device))
            prompt,*_=self.model.length_regulator(sem,ylens=torch.tensor([mel.shape[2]],device=self.device),n_quantizers=3,f0=None)
            result=(prompt,mel,style); path.parent.mkdir(parents=True,exist_ok=True)
            import tempfile, os
            fd,temp=tempfile.mkstemp(dir=path.parent,suffix='.npz'); os.close(fd)
            try:
                np.savez(temp,prompt=prompt.cpu().numpy(),mel=mel.cpu().numpy(),style=style.cpu().numpy())
                os.replace(temp,path)
            finally: Path(temp).unlink(missing_ok=True)
        self.refs={key:result} # bound memory: only current profile retained
        return result
    def convert(self, wave, profile, steps=20, check_cancel=None):
        if check_cancel is not None: check_cancel()
        if not 1<=steps<=100: raise ValueError('Steps must be 1–100')
        wave=np.asarray(wave,dtype=np.float32)
        if len(wave)<1024: wave=np.pad(wave,(0,1024-len(wave)))
        if len(wave)>self.sample_rate*20: raise ValueError('Adapter requires chunks of at most 20 seconds')
        if not np.isfinite(wave).all(): raise ValueError('Invalid model input')
        if np.max(np.abs(wave))<1e-6: return np.zeros_like(wave)
        import torch
        with self.lock, torch.inference_mode():
            for attempt in range(4):
                try:
                    if check_cancel is not None: check_cancel()
                    self.load()
                    if check_cancel is not None: check_cancel()
                    with Timer(self.device) as timer:
                        out=self._convert(wave,profile,steps,check_cancel=check_cancel)
                    if not np.isfinite(out).all(): raise FloatingPointError('NaN/Inf in model output')
                    self.stats.update(inference_seconds=timer.seconds,rtf=timer.seconds/(len(wave)/self.sample_rate),
                        backend=self.device,precision=self.precision,model_kind=self.model_kind)
                    if self.device=='xpu': self.stats['peak_allocated_bytes']=torch.xpu.max_memory_allocated()
                    if check_cancel is not None: check_cancel()
                    return out
                except Cancelled:
                    # Do not treat a cancellation as an inference error and retry
                    # the entire chunk on FP32 or CPU.
                    raise
                except (RuntimeError,FloatingPointError,NotImplementedError) as exc:
                    if self.compile_model:
                        self.unload(); self.compile_model=False
                        self.stats['fallbacks'].append('Disable compile after runtime failure: '+str(exc)); self.load(); continue
                    if self.precision!='fp32':
                        self.precision='fp32'; self.stats['fallbacks'].append('Retry FP32: '+str(exc)); continue
                    if self.device=='xpu':
                        self.unload(); self.device='cpu'; self.compile_model=False
                        self.stats['fallbacks'].append('Retry CPU after XPU error: '+str(exc)); self.load(); continue
                    raise
            raise RuntimeError('All safe inference retries failed')
    def _convert(self,wave,profile,steps,check_cancel=None):
        import torch
        if check_cancel is not None: check_cancel()
        prompt,mel2,style=self.reference(profile)
        if check_cancel is not None: check_cancel()
        w16=torch.from_numpy(resample(wave,self.sample_rate,16000)).unsqueeze(0).to(self.device)
        sem=self.semantic(w16)
        if check_cancel is not None: check_cancel()
        source=torch.from_numpy(wave).unsqueeze(0).to(self.device)
        mel=self.mel(source.float())
        cond,*_=self.model.length_regulator(sem,ylens=torch.tensor([mel.shape[2]],device=self.device),n_quantizers=3,f0=None)
        # Upstream context limit is 30s including the reference. Split cond to avoid overflow.
        window=30*self.sample_rate//256-mel2.shape[2]; overlap=16; chunks=[]; pos=0; previous=None
        if window<=overlap: raise ValueError('Reference leaves insufficient source context')
        dtype={'fp16':torch.float16,'bf16':torch.bfloat16}.get(self.precision)
        while pos<cond.shape[1]:
            if check_cancel is not None: check_cancel()
            part=cond[:,pos:pos+window]; last=pos+window>=cond.shape[1]
            cat=torch.cat((prompt,part),dim=1)
            context=torch.autocast(self.device,dtype=dtype) if dtype else contextlib.nullcontext()
            with context:
                target=self.model.cfm.inference(cat,torch.tensor([cat.shape[1]],device=self.device),
                    mel2,style,None,steps,inference_cfg_rate=.7,cancel_check=check_cancel)[:,:,mel2.shape[-1]:]
            out=self.vocoder(target.float()).reshape(-1).cpu().numpy()
            if previous is not None:
                n=min(len(previous),len(out)); fade=np.linspace(0,1,n,dtype=np.float32)
                out[:n]=previous[:n]*(1-fade)+out[:n]*fade
            if last: chunks.append(out); break
            n=overlap*256
            if len(out)<=n: raise RuntimeError('Model returned insufficient context')
            chunks.append(out[:-n]); previous=out[-n:]; pos+=target.shape[-1]-overlap
        return np.concatenate(chunks)
