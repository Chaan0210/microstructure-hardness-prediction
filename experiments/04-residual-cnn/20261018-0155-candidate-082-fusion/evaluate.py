"""Evaluate only after every nested fusion/control selection is frozen."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import engine as e
from fit_fusion import cohorts
import json,sys,csv,time
import numpy as np
def rm(p,y):return float(np.sqrt(np.mean((p-y)**2)))
def stats(base,p,y,fold,hi):
 a=(base-y)**2;b=(p-y)**2;g=a-b;rng=np.random.default_rng(2026101809);ix=rng.integers(0,len(y),(10000,len(y)));delta=np.sqrt(a[ix].mean(1))-np.sqrt(b[ix].mean(1));h=np.flatnonzero(hi);ih=h[rng.integers(0,len(h),(10000,len(h)))];dh=np.sqrt(a[ih].mean(1))-np.sqrt(b[ih].mean(1));order=np.argsort(g)[::-1];ff=[rm(base[fold==k],y[fold==k])-rm(p[fold==k],y[fold==k]) for k in range(5)];drop={str(n):rm(base[order[n:]],y[order[n:]])-rm(p[order[n:]],y[order[n:]]) for n in [5,10,20]};D=p[hi]-base[hi];X=np.c_[np.ones(hi.sum()),base[hi]];aff=X@np.linalg.lstsq(X,D,rcond=None)[0]
 return dict(overall_RMSE=rm(p,y),high_RMSE=rm(p[hi],y[hi]),gain=rm(base,y)-rm(p,y),high_gain=rm(base[hi],y[hi])-rm(p[hi],y[hi]),CI95=np.quantile(delta,[.025,.975]).tolist(),high_CI95=np.quantile(dh,[.025,.975]).tolist(),fold_gains=ff,improved_folds=sum(v>1e-10 for v in ff),removed_top=drop,top10_share=float(g[order[:10]].sum()/g.sum()) if abs(g.sum())>1e-12 else 0.,correction_RMS=float(np.sqrt(np.mean((p-base)**2))),affine_only_high_gain=rm(base[hi],y[hi])-rm(base[hi]+aff,y[hi]),nonaffine_only_high_gain=rm(base[hi],y[hi])-rm(base[hi]+D-aff,y[hi]))
def main():
 start=time.monotonic();assert json.loads((c.HERE/'fits-complete.json').read_text())['complete'];assert json.loads((c.HERE/'control-complete.json').read_text())['complete'];z=np.load(_repo_io(c.HERE/'reference.npz'));y=z['y'];base=z['base082'];hi=z['high'];ff=z['fold'];names=['primary','fixed','shared','global_','conditional','none','quality_calibration'];pred={name:np.full(400,np.nan) for name in names};components=np.full((400,5),np.nan);weights=np.array([.5,.125,.125,.125,.125]);selection=[]
 for k in range(5):
  p=np.load(_repo_io(c.HERE/f'fold{k}-predictions.npz'));ix=p['ix'];row=json.loads((c.HERE/f'fold{k}-selection.json').read_text());B,V,_,ref,beta=cohorts(k);act=e.high(V)
  for name in names[:-1]:pred[name][ix]=p[name]
  pred['quality_calibration'][ix]=np.load(_repo_io(c.HERE/f'control/fold{k}.npz'))['prediction'];fixed=row['models'][str(row['selected']['fixed'])];comp=np.tile(ref[:,None],(1,5));comp[act,1:4]=V['p76'][act,None]+beta*V['R'][act,1:4];comp[act,4]+=fixed['topology_beta']*V['R'][act,4];assert np.max(abs(comp@weights-p['fixed']))<1e-10;components[ix]=comp;selection.append(dict(fold=k,configuration=row['configs'][row['selected']['primary']],n_high=len(row['high_rows'])))
 results={name:stats(base,p,y,ff,hi) for name,p in pred.items()};assert np.array_equal(pred['none'],base);assert all(np.array_equal(p[~hi],base[~hi]) for p in pred.values());p=results['primary'];checks=dict(gain02=p['gain']>=.02,high=p['high_gain']>0,three_folds=p['improved_folds']>=3,CI95=p['CI95'][0]>0,top10=p['removed_top']['10']>0);cross={name:stats(pred[name],pred['primary'],y,ff,hi) for name in ['shared','quality_calibration','fixed']};weighted_mse=float(np.mean(((components-y[:,None])**2)@weights));disagreement=float(np.mean(((components-pred['fixed'][:,None])**2)@weights));mse=rm(pred['fixed'],y)**2;assert abs(weighted_mse-disagreement-mse)<1e-10
 c.save(_repo_io('results.json'),dict(baseline=dict(overall=rm(base,y),high=rm(base[hi],y[hi])),results=results,primary_vs_controls=cross,selection=selection,fixed_diversity=dict(weighted_individual_MSE=weighted_mse,disagreement_MSE=disagreement,mixture_MSE=mse,identity_maxdiff=abs(weighted_mse-disagreement-mse)),seconds=time.monotonic()-start,code_sha256=c.sha(__file__),plan_sha256=c.sha(c.HERE/'PLAN.md'),limitation='Reused400labels; fixed old representations/family choices; overlapping training scopes. Conditional bootstrap is not independent confirmation.'))
 c.save(_repo_io('screen-decision.json'),dict(advance=all(checks.values()),checks=checks,primary='nested selected procedure',new_information_supported=cross['quality_calibration']['CI95'][0]>0,official_best_changed=False,submitted=False));c.arrays('oof.npz',ids=z['ids'],y=y,fold=ff,high=hi,base082=base,**pred)
 with (c.HERE/'comparison.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['arm','overall_RMSE','high_RMSE','gain','CI95_low','CI95_high','improved_folds','top10_removed_gain'])
  for name,r in results.items():w.writerow([name,r['overall_RMSE'],r['high_RMSE'],r['gain'],*r['CI95'],r['improved_folds'],r['removed_top']['10']])
 with (c.HERE/'per-image.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','fold','high','hardness','082',*names,'primary_SSE_gain'])
  for j in range(400):w.writerow([z['ids'][j],ff[j],hi[j],y[j],base[j],*[pred[n][j] for n in names],(base[j]-y[j])**2-(pred['primary'][j]-y[j])**2])
 print((c.HERE/'comparison.csv').read_text());print('ADVANCE',all(checks.values()),checks,flush=True)
if __name__=='__main__':sys.addaudithook(c.guard);main()
