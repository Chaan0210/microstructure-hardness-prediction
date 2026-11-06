from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import time,sys,os
import numpy as np
import runtime as f
import features as w
R=f.R;P=f.P

def main():
 plan=f.check();assert f.read(R/'synthetic-validation.json')['passed'];allowed={P/'data/train'/f'{i}.png' for i in plan['IDs']}
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=f.Path(os.fsdecode(args[0])).absolute()
   if p.is_relative_to(P/'data') and p not in allowed:raise RuntimeError('Holdout/test/rawCSV forbidden')
 sys.addaudithook(guard);start=time.perf_counter();rows=[];hashes={};smoke={}
 with f.Monitor('feature-extraction',plan['resources']):
  for j,i in enumerate(plan['IDs']):
   with f.Image.open(_repo_io(P/'data/train'/f'{i}.png')) as im:assert im.mode=='L' and im.size==(256,256);x=np.asarray(im,dtype=float)/255
   a=np.array(w.extract(x));rows.append(a);hashes[i]=f.sha(P/'data/train'/f'{i}.png')
   if i in plan['smoke_IDs']:
    other=np.array(w.extract(x,backend='scipy'));error=float(abs(a-other).max());assert error<1e-10;smoke[i]=error
   if (j+1)%40==0:f.emit({'rows':j+1,'seconds':time.perf_counter()-start})
  X=np.stack(rows);assert X.shape==(400,2,15) and np.isfinite(X).all();assert hashes==f.read(P/'experiments/dinol-177/extraction.json')['image_sha256'];assert len(smoke)==3
  with (R/'features.npz').open('xb') as fp:np.savez_compressed(_repo_io(fp),ids=plan['IDs'],WPH=X[:,0],AMP=X[:,1])
 rec={'passed':True,'rows':400,'shape_perarm':[400,15],'raw400_image_hashes_exact':True,'NumPy_SciPy_smoke_errors':smoke,'feature_sha256':f.sha(R/'features.npz'),'seconds':time.perf_counter()-start,'noholdout_test':True,'no_fit_for_extraction':True,'image_sha256':hashes};f.write(R/'feature-validation.json',rec);f.emit({k:v for k,v in rec.items() if k!='image_sha256'});f.check()
if __name__=='__main__':
 try:main()
 except Exception as e:f.write(R/'extraction-failure.json',{'error':repr(e)});raise
