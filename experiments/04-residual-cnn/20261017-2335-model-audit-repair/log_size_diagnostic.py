"""Post-failure diagnostic: compare correction at matched256-source log contexts.

No new models, selection or candidate. Average squared errors by source across
five contexts; do not count repeated predictions as independent images.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import json,sys,time
import numpy as np,joblib
from threadpoolctl import threadpool_limits
def main():
 start=time.monotonic();d=c.inputs();base=np.load(_repo_io(c.OLD/'baseline076.npz'));ref=np.load(_repo_io(c.EXP/'highnoise-relational-20261016/inputs.npz'));X={};variants=['u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z']
 for v in variants:
  z=np.load(_repo_io(c.EXP/f'push-800/feat-{v}-anghcor-train.npz'),allow_pickle=True);pos={str(i):j for j,i in enumerate(z['ids'])};X[v]=z['X'][[pos[str(i)] for i in d['ids']]].astype(float)
 pp={a:np.tile(ref['base082'],(5,1)) for a in ['raw256','smear256','offset256']};rowpos={int(v):j for j,v in enumerate(d['dev'])};ns=[]
 for k,g in enumerate(d['graphs']):
  root=c.HERE/f'log-mean/fold{k}';z=np.load(_repo_io(root/'predictions.npz'));q=z['query'];ix=np.array([rowpos[int(v)] for v in q]);fac=float(z['factor']);off=float(z['bias'])
  for j in range(5):
   logs=[]
   for v in variants:
    p=root/f'{v}-inner{j}';m=joblib.load(_repo_io(p/'model.joblib'));ct=np.load(_repo_io(p/'context.npz'));assert len(ct['train'])==256 and set(ct['train']).isdisjoint(q);logs.append(m.predict(X[v][q]));ns.append(len(ct['train']))
   raw=np.exp(np.mean(logs,0))
   for a,values in [('raw256',raw),('smear256',raw*fac),('offset256',raw+off)]:pp[a][j,ix]+=.128*(values-base['logridge076'][ix])
 y=d['y'][d['dev']];errors={a:((p-y)**2).mean(0) for a,p in pp.items()};high=base['noise']>=12;rng=np.random.default_rng(2026101803);ii=rng.integers(0,400,(10000,400));out={}
 for a in ['smear256','offset256']:
  dist=np.sqrt(errors['raw256'][ii].mean(1))-np.sqrt(errors[a][ii].mean(1));out[a]=dict(gain_over_same_size_raw=float(np.sqrt(errors['raw256'].mean())-np.sqrt(errors[a].mean())),high_gain=float(np.sqrt(errors['raw256'][high].mean())-np.sqrt(errors[a][high].mean())),ci95=np.quantile(dist,[.025,.975]).tolist())
 c.arrays('log-matched-size-predictions.npz',ids=d['ids'][d['dev']],y=y,**pp)
 c.save(_repo_io('log-matched-size-diagnostic.json'),dict(only_diagnostic=True,new_fits=0,train_sizes=sorted(set(ns)),repeated_contexts_per_source=5,aggregation='Mean squared error per original source before bootstrap; not prediction averaging',results=out,seconds=time.monotonic()-start));print(json.dumps(out,indent=2))
if __name__=='__main__':
 sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):main()
