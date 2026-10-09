import threading
import numpy as np

class RingBuffer:
    """Bounded ring. Locks cover only memory copies, never inference or disk IO."""
    def __init__(self, capacity):
        self.data=np.zeros(capacity,dtype=np.float32); self.capacity=capacity
        self.read_pos=0; self.write_pos=0; self.size=0; self.dropped=0; self.lock=threading.Lock()
    def write(self, wave):
        wave=np.asarray(wave,dtype=np.float32).reshape(-1)
        with self.lock:
            if len(wave)>self.capacity:
                self.dropped+=len(wave)-self.capacity; wave=wave[-self.capacity:]
            overflow=max(0,self.size+len(wave)-self.capacity)
            self.read_pos=(self.read_pos+overflow)%self.capacity; self.size-=overflow; self.dropped+=overflow
            n=min(len(wave),self.capacity-self.write_pos)
            self.data[self.write_pos:self.write_pos+n]=wave[:n]; self.data[:len(wave)-n]=wave[n:]
            self.write_pos=(self.write_pos+len(wave))%self.capacity; self.size+=len(wave)
    def read(self,count, pad=True):
        with self.lock:
            n=min(count,self.size); first=min(n,self.capacity-self.read_pos)
            out=np.zeros(count if pad else n,dtype=np.float32)
            out[:first]=self.data[self.read_pos:self.read_pos+first]; out[first:n]=self.data[:n-first]
            self.read_pos=(self.read_pos+n)%self.capacity; self.size-=n; return out
    def available(self):
        with self.lock: return self.size
