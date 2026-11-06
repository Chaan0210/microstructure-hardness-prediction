"""Train the complete nested selection procedure without loading query labels."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import engine as e
import sys,json,time,resource
import numpy as np
from threadpoolctl import threadpool_limits
def cohorts(k,labels=True):
 z=np.load(_repo_io(c.HERE/f'inputs/fold{k}.npz'));B=dict(R=z['RB'],p76=z['p76B'],p73=z['p73B'],quality=z['qualityB'],noise=z['noiseB'],eligible=z['eligibleB'],ids=z['idsB']);B['y']=z['yB'];V=dict(R=z['RV'],p76=z['p76V'],quality=z['qualityV'],noise=z['noiseV'],eligible=z['eligibleV'],ids=z['idsV']);return B,V,z['ix'],z['p82V'],float(z['old_beta'])
def main():
 start=time.monotonic();c.save(_repo_io('fit-code-lock.json'),dict(engine=c.sha(c.HERE/'engine.py'),fit=c.sha(__file__),plan=c.sha(c.HERE/'PLAN.md'),common=c.sha(c.HERE/'common.py'),input_provenance=c.sha(c.HERE/'input-provenance.json')));records=[]
 for k in range(5):
  B,V,ix,p82,beta=cohorts(k);result=e.select_all(B);assert abs(result['full_beta']-beta)<1e-12;out={}
  for name,j in result['selected'].items():out[name]=e.predict(result['models'][str(j)],V)
  assert np.max(abs(out['none']-p82))<1e-10
  c.arrays(f'fold{k}-predictions.npz',ix=ix,ids=V['ids'],**out);c.save(_repo_io(f'fold{k}-selection.json'),result);records.append(dict(fold=k,selected={a:result['configs'][j] for a,j in result['selected'].items()},n_high=len(result['high_rows'])));assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*2**30
  print('FUSION FIT',k,records[-1],'seconds',round(time.monotonic()-start,1),flush=True);assert time.monotonic()-start<900,'15min screen-fit budget'
 c.save(_repo_io('fits-complete.json'),dict(complete=True,seconds=time.monotonic()-start,records=records,new_image_model_fits=0,new_meta_selection='LOO onB, including all incumbent/family/quality nuisance fits'))
if __name__=='__main__':
 sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):main()
