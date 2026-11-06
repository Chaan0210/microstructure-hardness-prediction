"""Inspect the frozen combination's weights and failure cases, without refitting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import engine as e
from fit_fusion import cohorts
import json,sys,csv
import numpy as np
def coefficients(model,q):
 h=e.high(q);out=np.zeros((len(q['R']),5));beta=model['beta'];kind=model['kind']
 if kind=='none':out[h,0]=beta
 elif kind=='fixed':out[h,:4]=beta*np.array([.625,.125,.125,.125]);out[h,4]=.125*model['topology_beta']
 else:
  theta=np.array(model['theta']);U=e.basis(q['quality'][h],np.array(model['quality_mean']),np.array(model['quality_scale']),kind=='conditional');out[np.ix_(h,np.arange(len(theta)))]=U@theta.T
 return out
def main():
 z=np.load(_repo_io(c.HERE/'oof.npz'));y=z['y'];p=z['primary'];base=z['base082'];weights=np.zeros((400,5));records=[]
 for k in range(5):
  B,V,ix,ref,beta=cohorts(k);row=json.loads((c.HERE/f'fold{k}-selection.json').read_text());model=row['models'][str(row['selected']['primary'])];w=coefficients(model,V);weights[ix]=w;manual=V['p76']+(w*V['R']).sum(1);assert np.max(abs(manual-p[ix]))<1e-10;h=e.high(V);records.append(dict(fold=k,configuration=row['configs'][row['selected']['primary']],incumbent_beta=beta,mean_high_coefficients=w[h].mean(0).tolist(),min_high_coefficients=w[h].min(0).tolist(),max_high_coefficients=w[h].max(0).tolist(),n_high=int(h.sum())))
 gain=(base-y)**2-(p-y)**2;order=np.argsort(gain);cases={name:[dict(ID=str(z['ids'][j]),fold=int(z['fold'][j]),hardness=float(y[j]),incumbent082=float(base[j]),fusion=float(p[j]),SSE_gain=float(gain[j])) for j in ix] for name,ix in [('harms',order[:10]),('helps',order[-10:][::-1])]}
 with (c.HERE/'coefficients.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','fold','high','oldCNN','076targetCNN','densePickCNN','densePoolCNN','topology_delta'])
  for j in range(400):w.writerow([z['ids'][j],z['fold'][j],bool(z['high'][j]),*weights[j]])
 c.save(_repo_io('diagnostics.json'),dict(coefficient_definition='Coefficients of residual corrections on top of076, not percentages of complete predictors.',folds=records,cases=cases,source_labels_used_only_for_posthoc_description=True));print(json.dumps(records,indent=2))
if __name__=='__main__':sys.addaudithook(c.guard);main()
