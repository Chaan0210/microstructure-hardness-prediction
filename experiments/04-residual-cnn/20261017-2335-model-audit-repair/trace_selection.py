"""Reproduce the pre-existing calibration target/selection audit, no new model."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import cnn_repair as cr
import evaluate as e
import sys,json
import numpy as np
def write_or_verify(name,value):
 p=c.HERE/name
 if p.exists():assert json.loads(p.read_text())==value,name
 else:c.save(_repo_io(name),value)
def main():
 d=c.inputs();fixed=json.loads((c.R4/'noise-calibration/summary.json').read_text())['aux']['selection'];trace=[];objectives=[]
 for k,g in enumerate(d['graphs']):
  old=np.load(_repo_io(c.R4/f'teachers/fold{k}-outer0/predictions.npz'));new=np.load(_repo_io(c.teacher_path(k,'outer0')));a=np.load(_repo_io(c.R4/f'anchors/fold{k}.npz'));q=old['query'];_,corr=e.corrections(k,'selector','old');hi=d['noise'][q]>=12;p73=.16*a['prediction_B']+.84*old['base'];p76=.16*a['prediction_B']+.84*new['base'];values={}
  for name,pred in [('073',p73),('076',p76)]:
   r=corr[hi];er=d['y'][q][hi]-pred[hi];values[name]=dict(raw_high_beta=float(r@er/(r@r)),error_correction_correlation=float(np.corrcoef(r,er)[0,1]))
  trace.append(dict(fold=k,n_high=int(hi.sum()),old082_high_weight=fixed[k]['addition']['weights'][1],old_pool_strength=fixed[k]['addition']['strength'],base_change_rms_high=float(np.sqrt(np.mean((p76[hi]-p73[hi])**2))),**values))
  good=np.nan_to_num(d['X'][q,list(d['keys']).index('n_obj')],nan=0)>=20;r=corr[good];target=(d['y'][q]-.16*a['prediction_B']-.84*old['base'])[good];band=d['noise'][q][good]>=12;strengths=['global',100,30,10,0];lo=[];ld=[]
  for s in strengths:
   pred=[]
   for j in range(len(r)):
    ix=np.arange(len(r))!=j;w=cr.coef_ns['fit_weights'](r[ix],target[ix],band[ix],s);pred.append(w[int(band[j])]*r[j])
   pred=np.array(pred);lo.append(float(np.mean((pred-target)**2)));pred[~band]=0;ld.append(float(np.mean((pred-target)**2)))
  objectives.append(dict(fold=k,n_high=int(band.sum()),strengths=strengths,original_loo_mse=lo,deployed_policy_loo_mse=ld,original_choice=strengths[int(np.argmin(lo))],aligned_choice=strengths[int(np.argmin(ld))]))
 write_or_verify('calibration-trace.json',trace);write_or_verify('selector-objective-audit.json',dict(descriptive_only=True,new_candidate=False,outer_labels_used=False,already_tested_remedy='076-complement/calibration_order.py; high-only LOO remedy previously failed influence/role checks',folds=objectives));print('Calibration trace and objective audit reproduced')
if __name__=='__main__':sys.addaudithook(c.guard);main()
