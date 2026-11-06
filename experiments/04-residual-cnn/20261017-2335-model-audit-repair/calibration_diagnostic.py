"""Descriptive sensitivity of a fixed internal coefficient procedure.

No candidate, hyperparameter search or outer-label fitting. All omission tests
operate on the existing B calibration partition only.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import cnn_repair as cr
import evaluate as e
import sys,time,json
import numpy as np
def main():
 start=time.monotonic();d=c.inputs();rows=[]
 for k,g in enumerate(d['graphs']):
  t=np.load(_repo_io(c.teacher_path(k,'outer0')));a=np.load(_repo_io(c.R4/f'anchors/fold{k}.npz'));B=t['query'];assert np.array_equal(B,a['B']);good=np.nan_to_num(d['X'][B,list(d['keys']).index('n_obj')],nan=0)>=20;Bgood=B[good];base=.16*a['prediction_B']+.84*t['base'];target=(d['y'][B]-base)[good];band=d['noise'][Bgood]>=12
  for arm in ['old','retarget']:
   q,p=e.corrections(k,'selector',arm);assert np.array_equal(q,B);p=p[good];full,meta=cr.select(p,target,band);weights=[];strengths=[]
   for j in range(len(p)):
    ix=np.arange(len(p))!=j;w,m=cr.select(p[ix],target[ix],band[ix]);weights.append(float(w[1]));strengths.append(str(m['strength']))
   w=np.array(weights);jj=int(np.argmax(abs(w-full[1])));rows.append(dict(fold=k,arm=arm,n_B=len(Bgood),n_high=int(band.sum()),full_high_weight=float(full[1]),full_strength=meta['strength'],delete_one_weight_min=float(w.min()),delete_one_weight_max=float(w.max()),delete_one_weight_std=float(w.std()),changed_strength_count=sum(v!=str(meta['strength']) for v in strengths),largest_change=float(abs(w-full[1]).max()),largest_change_removed_id=str(d['ids'][Bgood[jj]]),ids=d['ids'][Bgood].tolist(),delete_one_weights=weights))
 c.save(_repo_io('calibration-delete-one.json'),dict(only_diagnostic=True,uses_outer_labels=False,rows=rows,seconds=time.monotonic()-start));print(json.dumps([{k:v for k,v in r.items() if k not in ['ids','delete_one_weights']} for r in rows],indent=2),flush=True)
if __name__=='__main__':sys.addaudithook(c.guard);main()
