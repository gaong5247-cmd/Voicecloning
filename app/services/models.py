"""Explicit downloads only; inference never invokes this module's downloader."""
import json, urllib.request, os, hashlib
from pathlib import Path
from app.config import DATA, ROOT, digest

SEED_FILES=['DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth','config_dit_mel_seed_uvit_whisper_small_wavenet.yml']
GROUPS={'voice':['Plachta/Seed-VC','funasr/campplus','openai/whisper-small','nvidia/bigvgan_v2_22khz_80band_256x'],
        'separation':['speechbrain/sepformer-wsj02mix','speechbrain/spkrec-ecapa-voxceleb'],
        'diarization':['pyannote/speaker-diarization-community-1']}

def model_dir(repo): return DATA/'models'/repo.replace('/','--')

def selected(repo, filename):
    if repo=='Plachta/Seed-VC': return filename in SEED_FILES
    if repo=='funasr/campplus': return filename=='campplus_cn_common.bin'
    if repo=='openai/whisper-small': return filename in ['config.json','preprocessor_config.json','model.safetensors']
    if repo.startswith('nvidia/'): return filename in ['config.json','bigvgan_generator.pt']
    if repo=='speechbrain/sepformer-wsj02mix': return filename in ['hyperparams.yaml','encoder.ckpt','decoder.ckpt','masknet.ckpt']
    if repo=='speechbrain/spkrec-ecapa-voxceleb': return filename in ['hyperparams.yaml','embedding_model.ckpt','classifier.ckpt','mean_var_norm_emb.ckpt','label_encoder.txt']
    return not filename.endswith(('.md','.gitattributes','.png','.jpg','.wav','.gif'))

def download_group(group, token=None, progress=lambda message:None, cancelled=lambda:None):
    lock=json.loads((ROOT/'config/models.lock.json').read_text())
    for repo in GROUPS[group]:
        item=lock[repo]; directory=model_dir(repo); directory.mkdir(parents=True,exist_ok=True)
        for name, expected in item['files'].items():
            if not selected(repo,name): continue
            cancelled(); dest=directory/name; dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and (not expected or digest(dest)==expected):
                progress('Cached: '+name); continue
            progress('Downloading: '+repo+'/'+name)
            url=f"https://huggingface.co/{repo}/resolve/{item['revision']}/{name}"
            headers={'User-Agent':'CloneVoiceStudio/0.1'}
            if token: headers['Authorization']='Bearer '+token
            request=urllib.request.Request(url,headers=headers)
            temp=dest.with_suffix(dest.suffix+'.partial'); h=hashlib.sha256()
            try:
                with urllib.request.urlopen(request,timeout=60) as response, temp.open('wb') as f:
                    while True:
                        cancelled(); block=response.read(1024*1024)
                        if not block: break
                        f.write(block); h.update(block)
                if expected and h.hexdigest()!=expected: raise RuntimeError('Model SHA256 mismatch: '+name)
                os.replace(temp,dest)
            finally: temp.unlink(missing_ok=True)
        (directory/'revision.txt').write_text(item['revision'])
    progress('Model download complete. No audio was uploaded.')

def check_group(group):
    lock=json.loads((ROOT/'config/models.lock.json').read_text())
    missing=[]
    for repo in GROUPS[group]:
        item=lock[repo]
        for name,expected in item['files'].items():
            if selected(repo,name):
                path=model_dir(repo)/name
                if not path.is_file(): missing.append(repo+'/'+name)
                elif expected and digest(path)!=expected: raise RuntimeError('Model integrity failed: '+name)
    if missing: raise RuntimeError('Download '+group+' models in Model Manager first: '+', '.join(missing[:4]))
