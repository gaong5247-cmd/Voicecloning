import threading, time
import numpy as np
from app.audio_io.ring import RingBuffer
from app.audio_io.files import resample, align_length

class Realtime:
    def __init__(self,engine,profile,input_device,output_device,sample_rate=48000,
                 chunk_seconds=1.0,steps=6,gate_db=-45,gain=1):
        if chunk_seconds<.5: raise ValueError('This V1 adapter requires >=0.5s blocks; tiny streaming model not yet integrated.')
        self.engine=engine; self.profile=profile; self.sr=sample_rate
        self.input_device=input_device; self.output_device=output_device
        self.chunk=int(sample_rate*chunk_seconds); self.steps=steps; self.gate=10**(gate_db/20); self.gain=gain
        self.input=RingBuffer(sample_rate*8); self.output=RingBuffer(sample_rate*8)
        self.stop_event=threading.Event(); self.muted=False; self.original=False
        self.metrics={'underruns':0,'callback_status_count':0,'model_seconds':None,
                      'hardware_end_to_end_ms':None,'latency_kind':'software estimate only','errors':[]}
        self.tail=np.zeros(0,dtype=np.float32)
    def start(self):
        import sounddevice as sd
        # Loading happens before opening devices, on the GUI worker.
        self.engine.warm_profile(self.profile)
        self.capture=sd.InputStream(device=self.input_device,samplerate=self.sr,channels=1,
                                    dtype='float32',blocksize=int(self.sr*.02),callback=self._input)
        self.playback=sd.OutputStream(device=self.output_device,samplerate=self.sr,channels=1,
                                    dtype='float32',blocksize=int(self.sr*.02),callback=self._output)
        try:
            self.capture.start(); self.playback.start()
            self.metrics['input_device_reported_seconds']=self.capture.latency
            self.metrics['output_device_reported_seconds']=self.playback.latency
            self.worker=threading.Thread(target=self._work,name='VCWorker',daemon=True); self.worker.start()
        except Exception:
            self.capture.close(); self.playback.close(); raise
    def _input(self,indata,frames,timing,status):
        if status: self.metrics['callback_status_count']+=1
        self.input.write(indata[:,0])
    def _output(self,outdata,frames,timing,status):
        if status: self.metrics['callback_status_count']+=1
        if self.output.available()<frames: self.metrics['underruns']+=1
        outdata[:,0]=self.output.read(frames)
    def _work(self):
        history=np.zeros(0,dtype=np.float32)
        while not self.stop_event.is_set():
            if self.input.available()<self.chunk: self.stop_event.wait(.005); continue
            block=self.input.read(self.chunk); start=time.perf_counter()
            try:
                rms=float(np.sqrt(np.mean(block**2))); self.metrics['voice_activity']=rms>self.gate
                if self.muted or rms<self.gate: out=np.zeros_like(block)
                elif self.original: out=block.copy()
                else:
                    context=np.concatenate((history,block)); mono=resample(context,self.sr,self.engine.sample_rate)
                    converted=self.engine.convert(mono,self.profile,self.steps)
                    converted=resample(converted,self.engine.sample_rate,self.sr)
                    converted,_=align_length(converted,len(context)); out=converted[-len(block):]
                history=block[-int(self.sr*.3):].copy()
                n=min(int(self.sr*.01),len(self.tail),len(out))
                if n:
                    fade=np.linspace(0,1,n); out[:n]=self.tail[-n:]*(1-fade)+out[:n]*fade
                self.tail=out[-int(self.sr*.01):].copy()
                self.output.write(np.clip(out*self.gain,-.98,.98))
                sec=time.perf_counter()-start
                self.metrics.update(model_seconds=sec,rtf=sec/(len(block)/self.sr),
                    input_queue_seconds=self.input.available()/self.sr,output_queue_seconds=self.output.available()/self.sr,
                    dropped_input_samples=self.input.dropped,dropped_output_samples=self.output.dropped)
                self.metrics['software_latency_estimate_ms']=1000*(len(block)/self.sr+sec+self.input.available()/self.sr+self.output.available()/self.sr+
                    self.metrics['input_device_reported_seconds']+self.metrics['output_device_reported_seconds'])
                if sec>len(block)/self.sr and self.steps>2: self.steps=max(2,self.steps-1)
            except Exception as exc:
                self.metrics['errors']=self.metrics['errors'][-9:]+[str(exc)]
                self.output.write(np.zeros_like(block)) # explicit mute on failed block; never impersonation via passthrough
    def stop(self):
        self.stop_event.set()
        for name in ('capture','playback'):
            stream=getattr(self,name,None)
            if stream: stream.stop(); stream.close()
        if getattr(self,'worker',None): self.worker.join(timeout=2)
