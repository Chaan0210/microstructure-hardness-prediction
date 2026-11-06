"""Inference-only frozen H300 flow cache for the sealed candidate; no learning."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];OUT=_repo_path(HERE, 'candidate-flow46-v1')
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME','CELLPOSE_LOCAL_MODELS_PATH']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HUB_OFFLINE']='1'
import csv,json,time,hashlib,gc,datetime
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
import domain_semantics as ds

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);r=json.loads((_repo_path(OUT, 'frozen-recipe.json')).read_text());assert r['status'].startswith('frozen_') and sha(_repo_path(HERE, 'domain_semantics.py'))==r['flow_source_sha256'];assert sha(_repo_path(ROOT, r['flow_model']))==r['flow_model_sha256'];assert (_repo_path(OUT, 'fit-complete.json')).exists()
 lock=_repo_path(OUT, 'implementation-lock.json');expected={name:sha(_repo_path(HERE, name)) for name in ['flow_candidate_finalize.py','flow_test_inference.py']}
 if lock.exists():assert json.loads(lock.read_text())['source_hashes']==expected
 else:lock.write_text(json.dumps(dict(source_hashes=expected,sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_test_image_access=True),indent=2))
 rows=list(csv.DictReader((_repo_path(ROOT, 'data/sample_submission.csv')).open()));ids=np.array([x['ID'] for x in rows]);assert len(ids)==1000 and len(set(ids))==1000;path=_repo_path(OUT, 'test-flow-features.npz')
 if path.exists():return
 part=_repo_path(OUT, 'test-flow-partial.npz');values=[];elapsed=0.
 if part.exists():
  z=np.load(_repo_io(part));values=list(z['flow']);assert np.array_equal(z['ids'],ids[:len(values)]);assert str(z['recipe_sha256'])==sha(_repo_path(OUT, 'frozen-recipe.json'));elapsed=float(z['elapsed'])
 net=ds.model('H300','cpu');capture={};handle=net.encoder.neck.register_forward_hook(lambda module,args,value:capture.update(neck=value));start=time.monotonic();first=len(values)
 try:
  with torch.no_grad():
   for ident in ids[first:]:
    with Image.open(_repo_path(ROOT, 'data/test', f'{ident}.png')) as image:assert image.mode=='L' and image.size==(256,256);raw=np.array(image)
    vv=[]
    for k in [0,2]:
     x=np.stack([np.ascontiguousarray(np.rot90(ds.normalize(raw),rot)) for rot in [k,k+1]])[:,None];flow,_=net(torch.tensor(x));neck=capture.pop('neck').numpy();flow=flow.numpy()
     for j,rot in enumerate([k,k+1]):_,f,keys=ds.descriptors(neck[j],flow[j],np.rot90(raw,rot));vv.append(f)
    f=np.mean(vv,0);assert f.shape==(46,) and np.isfinite(f).all();values.append(f)
    if len(values)%10==0:
     np.savez(_repo_path(OUT, 'test-flow-partial-new.npz'),ids=ids[:len(values)],flow=np.array(values),recipe_sha256=sha(_repo_path(OUT, 'frozen-recipe.json')),elapsed=elapsed+time.monotonic()-start);(_repo_path(OUT, 'test-flow-partial-new.npz')).replace(part);print('INFERENCE_ROWS',len(values),'seconds',elapsed+time.monotonic()-start,flush=True)
 finally:handle.remove()
 assert keys==r['flow_keys'];np.savez(_repo_io(path),ids=ids,flow=np.array(values),flow_keys=np.array(keys));(_repo_path(OUT, 'test-flow-complete.json')).write_text(json.dumps(dict(complete=True,rows=1000,seconds=elapsed+time.monotonic()-start,recipe_sha256=sha(_repo_path(OUT, 'frozen-recipe.json')),source_hashes=expected,inference_only=True,parameters_updated=0,labels_read=False),indent=2));print('FLOW_INFERENCE_COMPLETE',flush=True)
if __name__=='__main__':
 with threadpool_limits(2):main()
