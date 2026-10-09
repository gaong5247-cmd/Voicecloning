import platform, time
import numpy as np
from app.config import DATA, atomic_json

def diagnose():
    result={'platform':platform.platform(),'xpu_available':False,'gpu_utilization':None,
            'hardware_end_to_end_latency_ms':None,'openvino':None,'npu':'not validated'}
    try:
        import torch
        result['torch']=torch.__version__
        result['torch_xpu_runtime']=getattr(torch.version,'xpu',None)
        result['xpu_wheel_installed']=result['torch_xpu_runtime'] is not None
        result['xpu_available']=torch.xpu.is_available()
        if not result['xpu_available']:
            result['xpu_diagnosis']=('CPU-only PyTorch wheel installed; use the XPU portable build' if not result['xpu_wheel_installed'] else 'XPU wheel present but no Intel XPU detected; check driver, hardware, and runtime DLLs')
        if result['xpu_available']:
            result['device']=torch.xpu.get_device_name(0)
            result['properties']=str(torch.xpu.get_device_properties(0))
            result['allocated_bytes']=torch.xpu.memory_allocated()
            result['precision_probes']={}
            for precision,dtype in [('fp32',torch.float32),('fp16',torch.float16),('bf16',torch.bfloat16)]:
                try:
                    a=torch.randn(32,32,device='xpu',dtype=dtype)
                    b=a@a; torch.xpu.synchronize()
                    result['precision_probes'][precision]=bool(torch.isfinite(b).all().item())
                except Exception as exc: result['precision_probes'][precision]=str(exc)
    except Exception as exc: result['torch_error']=str(exc)
    try:
        import openvino as ov
        result['openvino']={'version':ov.__version__,'devices':ov.Core().available_devices,
                             'voice_conversion':'no validated converted model'}
    except ImportError: pass
    atomic_json(DATA/'diagnostics.json',result); return result

def select_device(requested='auto'):
    import torch
    if requested not in ('auto','cpu','xpu'): raise ValueError('Unsupported execution backend')
    if requested=='xpu' and not torch.xpu.is_available():
        if getattr(torch.version,'xpu',None) is None:
            raise RuntimeError('XPU unavailable: this executable contains a CPU-only PyTorch wheel. Install the XPU portable build.')
        raise RuntimeError('XPU unavailable: XPU PyTorch is installed but no Intel GPU is detected. Check the Intel graphics driver and runtime.')
    return 'xpu' if requested!='cpu' and torch.xpu.is_available() else 'cpu'

class Timer:
    def __init__(self, device): self.device=device
    def sync(self):
        if self.device=='xpu':
            import torch; torch.xpu.synchronize()
    def __enter__(self): self.sync(); self.start=time.perf_counter(); return self
    def __exit__(self,*args): self.sync(); self.seconds=time.perf_counter()-self.start
