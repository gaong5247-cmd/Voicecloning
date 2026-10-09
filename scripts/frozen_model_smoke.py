"""Run the packaged program, including explicit downloads, with licensed synthetic fixtures."""
import subprocess, os, sys, json, time, wave
from pathlib import Path

root=Path(__file__).resolve().parents[1]
exe=Path(sys.argv[1]).resolve(); destination=Path(sys.argv[2]).resolve()
destination.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,CLONEVOICE_DATA=str(destination),CLONEVOICE_CPU_THREADS='4')

def run(*args,timeout=1800):
    result=subprocess.run([str(exe),*map(str,args)],env=env,timeout=timeout)
    if result.returncode:
        for p in (destination/'logs').glob('*.log'):
            print(p.name, p.read_text(encoding='utf-8',errors='replace')[-12000:])
        raise RuntimeError(f'Frozen command failed ({result.returncode}): {args[0]}')

start=time.perf_counter()
run('--download','voice')
run('--reference',root/'tests/fixtures/reference.wav','--profile','CI Synthetic Voice','--consent')
profiles=list((destination/'VoiceProfiles').glob('*/manifest.json'))
profile=json.loads(profiles[-1].read_text(encoding='utf-8'))
source=root/'tests/fixtures/source.wav'; output=destination/'converted.wav'
run('--source',source,'--profile',profile['id'],'--output',output,'--backend','cpu','--steps','6')
with wave.open(str(source)) as wav: duration=wav.getnframes()/wav.getframerate()
with wave.open(str(output)) as wav:
    frames=wav.getnframes();sr=wav.getframerate();samples=wav.readframes(frames)
    assert sr==22050 and abs(frames/sr-duration)<1/sr+.0001, 'Output timing mismatch'
    assert any(samples), 'Output is silent'
session=max((destination/'sessions').glob('*/session.json'),key=lambda p:p.stat().st_mtime)
record=json.loads(session.read_text(encoding='utf-8'))
assert record['state']=='complete'
report={'status':'passed','frozen_executable':exe.name,'runtime':'Windows' if os.name=='nt' else 'Linux',
        'input_seconds':duration,'output_seconds':frames/sr,'including_download_seconds':time.perf_counter()-start,
        'metrics':record['metrics'],'scope':'actual packaged CPU model conversion on synthetic English; no XPU/quality claim'}
Path('validation').mkdir(exist_ok=True)
Path('validation/frozen-model.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
