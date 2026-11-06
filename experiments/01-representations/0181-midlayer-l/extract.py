from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import time,sys,os
import numpy as np,torch
import runtime as f
R=f.R;P=f.P

def feature(m,x):
 with torch.inference_mode():
  layers=m.get_intermediate_layers(x,n=[11,23],return_class_token=True,norm=True);assert all(p.shape==(1,1296,1024) and c.shape==(1,1024) for p,c in layers)
  z=[torch.cat([c,p.mean(1),p.std(1,correction=0)],1).cpu().numpy()[0] for p,c in layers]
 return np.stack(z)
def main():
 plan=f.check();allowed={P/'data/train'/f'{i}.png' for i in plan['IDs']}
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=f.Path(os.fsdecode(args[0])).absolute()
   if p.is_relative_to(P/'data') and p not in allowed:raise RuntimeError('Holdout/test/rawCSV forbidden')
 sys.addaudithook(guard);start=time.perf_counter();rows=[];hashes={};smoke={}
 with f.Monitor('feature-extraction',plan['resources']):
  m=f.model();initial=f.statehash(m)
  for device in ['cpu','mps']:
   m=m.to(device);smoke[device]=np.stack([feature(m,torch.from_numpy(f.image_input(i)[None]).to(device)) for i in plan['smoke_IDs']])
  drift=float(abs(smoke['cpu']-smoke['mps']).max());assert drift<.002
  with np.load(_repo_io(P/'experiments/dinol-177/features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==plan['IDs'];old=np.stack([z['X'][:,:3072],z['X'][:,6144:]],axis=1)
  for j,i in enumerate(plan['IDs']):
   a=feature(m,torch.from_numpy(f.image_input(i)[None]).to('mps'));np.testing.assert_array_equal(a,old[j]);rows.append(a);hashes[i]=f.sha(P/'data/train'/f'{i}.png')
   if (j+1)%80==0:f.emit({'rows':j+1,'seconds':time.perf_counter()-start})
  X=np.stack(rows).astype(float);assert X.shape==(400,2,3072) and np.isfinite(X).all();assert initial==f.statehash(m);assert all(p.grad is None for p in m.parameters());assert hashes==f.read(P/'experiments/dinol-177/extraction.json')['image_sha256']
  for j,i in enumerate(plan['smoke_IDs']):np.testing.assert_array_equal(X[plan['IDs'].index(i)],smoke['mps'][j])
  del m;torch.mps.empty_cache()
  with (R/'features.npz').open('xb') as fp:np.savez_compressed(_repo_io(fp),ids=plan['IDs'],MID=X[:,0],LAST=X[:,1])
 rec={'passed':True,'rows':400,'shape_perlayer':[400,3072],'CPU_MPS_smoke_max_error':drift,'model_state_unchanged':True,'fresh400_exact177_cached_layers':True,'raw400_image_hashes_exact':True,'smoke_repeat_exact':True,'feature_sha256':f.sha(R/'features.npz'),'seconds':time.perf_counter()-start,'noholdout_test':True};f.write(R/'feature-validation.json',rec);f.emit(rec)
if __name__=='__main__':
 try:main()
 except Exception as e:f.write(R/'extraction-failure.json',{'error':repr(e)});raise
