"""Pinned original DINOv2 ViT-S/14, frozen local development-only features."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,hashlib,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
os.environ['TORCH_HOME']=str(_repo_path(ROOT, '.torch-cache'));os.environ['XDG_CACHE_HOME']=str(_repo_path(ROOT, '.cache'));os.environ['HF_HOME']=str(_repo_path(ROOT, '.hf-cache'));os.environ['XFORMERS_DISABLED']='1'
import numpy as np
from PIL import Image
import torch
VENDOR=_repo_path(ROOT, 'vendor/dinov2-official');PROVENANCE=json.loads((_repo_path(VENDOR, 'provenance.json')).read_text());SOURCE=_repo_path(ROOT, PROVENANCE['source_dir']);sys.path.insert(0,str(SOURCE))
from dinov2.hub.backbones import dinov2_vits14
OUT=_repo_path(ROOT, 'artifacts/dino-frozen-037')
MEAN=np.array([.485,.456,.406],dtype=np.float32)[:,None,None];STD=np.array([.229,.224,.225],dtype=np.float32)[:,None,None]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,o):
 with p.open('x') as f:json.dump(o,f,indent=2,allow_nan=False)
def model():
 w=_repo_path(VENDOR, 'dinov2_vits14_pretrain.pth');assert sha(w)==PROVENANCE[w.name]['sha256']
 m=dinov2_vits14(pretrained=False);m.load_state_dict(torch.load(_repo_io(w),map_location='cpu',weights_only=True),strict=True);m.requires_grad_(False);m.eval();assert not any(p.requires_grad for p in m.parameters());assert m.num_register_tokens==0 and m.embed_dim==384 and m.patch_size==14
 return m
def preprocess(ident):
 with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:
  assert im.mode=='L' and im.size==(256,256)
  a=np.asarray(im.resize((252,252),Image.Resampling.BICUBIC),dtype=np.float32)/255
 return np.ascontiguousarray((np.repeat(a[None],3,axis=0)-MEAN)/STD)
def descriptor(m,x):
 with torch.inference_mode():
  a=m.forward_features(x);c=a['x_norm_clstoken'];p=a['x_norm_patchtokens'];assert c.shape[1]==384 and p.shape[1:]==(324,384)
  out=torch.cat((c,p.mean(1),p.std(1,correction=0)),1)
 assert torch.isfinite(out).all();return out

def scope():
 sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());ids=sorted(sp['development_folds']);assert len(ids)==400 and not set(ids)&set(sp['holdout']);return ids,sp

def smoke():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(2);ids,sp=scope();smoke_ids=[i for i in ids if sp['development_folds'][i]!=0][:4];x=torch.from_numpy(np.stack([preprocess(i) for i in smoke_ids]));m=model();results={};outputs={}
 for device in ('cpu','mps'):
  if device=='mps' and not torch.backends.mps.is_available():continue
  m=m.to(device);xb=x.to(device);times=[]
  for k in range(4):
   if device=='mps':torch.mps.synchronize()
   t=time.perf_counter();p=descriptor(m,xb)
   if device=='mps':torch.mps.synchronize()
   if k:times.append(time.perf_counter()-t)
  outputs[device]=p.cpu().numpy();results[device]={'median_batch4_seconds':float(np.median(times)),'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
  if device=='mps':results[device]['driver_bytes']=torch.mps.driver_allocated_memory()
 drift=float(abs(outputs['cpu']-outputs['mps']).max()) if 'mps' in outputs else None
 if drift is not None:np.testing.assert_allclose(outputs['cpu'],outputs['mps'],rtol=2e-3,atol=2e-3)
 assert not m.training and all(p.grad is None for p in m.parameters())
 sources={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.glob('dinov2/**/*.py') if '/cell_dino/' not in str(p)}
 report={'model':'dinov2_vits14 original non-register','parameters':sum(p.numel() for p in m.parameters()),'torch':torch.__version__,'license':'Apache-2.0:officialREADME License section andMODEL_CARD License','source':PROVENANCE,'official_source_sha256':sources,'metadata_sha256':{n:sha(SOURCE/n) for n in ('LICENSE','README.md','MODEL_CARD.md')},'preprocessing':'PIL bicubic full256→252,no crop;grayscale RGB replication,[0,1],ImageNetmean(.485,.456,.406)/std(.229,.224,.225) matching officialtransforms.py','features':'CLS384+patchmean384+patchpopulationstd384 from final normalized tokens;324patches','devices':results,'cpu_mps_max_abs_feature_difference':drift,'frozen_eval_and_no_grad':True,'smoke_ids':smoke_ids,'scope':'4originalfold0trainrows only;no labels consumed','dependency_installation':'Vendored pinned official source, loaded directly in existing uv.venv. Existingtorch/numpy/Pillow sufficient; no additional package installed.','source_sha256':sha(Path(__file__))};write(_repo_path(OUT, 'smoke.json'),report);print(json.dumps(report|{'official_source_sha256':'saved in artifact'}),flush=True)

def extract():
 report=json.loads((_repo_path(OUT, 'smoke.json')).read_text());assert sha(Path(__file__))==report['source_sha256']
 for n,h in report['official_source_sha256'].items():assert sha(SOURCE/n)==h
 ids,sp=scope();device='mps' if 'mps' in report['devices'] else 'cpu';torch.set_num_threads(2);m=model().to(device);initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};features=[];start=time.perf_counter()
 for j in range(0,len(ids),8):
  x=torch.from_numpy(np.stack([preprocess(i) for i in ids[j:j+8]])).to(device);features.append(descriptor(m,x).cpu().numpy())
  if (j+8)%80==0:print(json.dumps({'completed':j+8,'seconds':time.perf_counter()-start}),flush=True)
 X=np.concatenate(features);assert X.shape==(400,1152) and np.isfinite(X).all()
 assert all(torch.equal(initial[k],v.detach().cpu()) for k,v in m.state_dict().items());assert all(p.grad is None for p in m.parameters())
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X)
 details={'device':device,'shape':list(X.shape),'seconds':time.perf_counter()-start,'weights_unchanged':True,'feature_sha256':sha(_repo_path(OUT, 'development_features.npz')),'source_sha256':sha(Path(__file__)),'image_sha256':{i:sha(_repo_path(ROOT, 'data/train', f'{i}.png')) for i in ids},'scope':'Frozen feature extraction from development400 only;no fitting,no target/holdout/test data','rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};write(_repo_path(OUT, 'extraction.json'),details);print(json.dumps(details|{'image_sha256':'saved in artifact'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','extract'));a=p.parse_args();smoke() if a.mode=='smoke' else extract()
