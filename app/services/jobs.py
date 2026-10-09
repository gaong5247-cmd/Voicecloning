import json, uuid, threading, time
from pathlib import Path
from app.config import DATA, atomic_json

class Cancelled(RuntimeError): pass
class Job:
    def __init__(self, source, profile, settings, root=None):
        self.id=uuid.uuid4().hex; self.path=Path(root or DATA/'sessions')/self.id
        self.path.mkdir(parents=True); self.cancelled=threading.Event()
        self.record={'id':self.id,'source':str(source),'profile':profile,'settings':settings,
                     'state':'pending','chunks':{},'errors':[],'created':time.time()}
        self.save()
    @classmethod
    def resume(cls, manifest):
        obj=cls.__new__(cls); obj.path=Path(manifest).parent
        obj.record=json.loads(Path(manifest).read_text(encoding='utf-8')); obj.id=obj.record['id']
        obj.cancelled=threading.Event(); return obj
    def save(self): atomic_json(self.path/'session.json',self.record)
    def check(self):
        if self.cancelled.is_set(): raise Cancelled('Task cancelled; finished chunks kept for resume.')
    def state(self, value): self.record['state']=value; self.save()
