"""Fixed paired decisions; controls are never promoted as outer-score winners."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import cnn_repair as cr
import sys,json,csv
import numpy as np

def rm(a,y):return float(np.sqrt(np.mean((a-y)**2)))
def compare(base,pred,y,fold,high):
 e0=(base-y)**2;e1=(pred-y)**2;gain=e0-e1;rng=np.random.default_rng(2026101801);idx=rng.integers(0,len(y),(10000,len(y)));dist=np.sqrt(e0[idx].mean(1))-np.sqrt(e1[idx].mean(1));hi=np.flatnonzero(high);idxh=hi[rng.integers(0,len(hi),(10000,len(hi)))];dh=np.sqrt(e0[idxh].mean(1))-np.sqrt(e1[idxh].mean(1));order=np.argsort(gain)[::-1];foldgain=[rm(base[fold==k],y[fold==k])-rm(pred[fold==k],y[fold==k]) for k in range(5)]
 out=dict(overall_rmse=rm(pred,y),high_rmse=rm(pred[high],y[high]),overall_gain=rm(base,y)-rm(pred,y),high_gain=rm(base[high],y[high])-rm(pred[high],y[high]),ci95=np.quantile(dist,[.025,.975]).tolist(),family_ci=np.quantile(dist,[.008333333333333333,.9916666666666667]).tolist(),high_ci95=np.quantile(dh,[.025,.975]).tolist(),fold_gains=foldgain,improved_folds=sum(x>1e-10 for x in foldgain),top5_removed_gain=rm(base[order[5:]],y[order[5:]])-rm(pred[order[5:]],y[order[5:]]),top10_removed_gain=rm(base[order[10:]],y[order[10:]])-rm(pred[order[10:]],y[order[10:]]),top10_share_net_SSE=float(gain[order[:10]].sum()/gain.sum()) if abs(gain.sum())>1e-12 else 0.,correction_rms=float(np.sqrt(np.mean((pred-base)**2))))
 # Decomposition is descriptive; no fitted deployment calibration is made.
 delta=pred[high]-base[high];X=np.c_[np.ones(high.sum()),base[high]];aff=X@np.linalg.lstsq(X,delta,rcond=None)[0];out['affine_only_high_gain']=rm(base[high],y[high])-rm(base[high]+aff,y[high]);out['nonaffine_only_high_gain']=rm(base[high],y[high])-rm(base[high]+delta-aff,y[high]);return out

def corrections(k,stage,kind,seed=None):
 pp=[]
 for s in ([seed] if seed is not None else [0,1]):
  if kind=='old':path=c.R4/f'aux-seed{s}/fold{k}-{stage}'
  elif kind=='retarget':
   path=c.HERE/f'cnn-retarget/seed{s}/fold{k}-{stage}'
   if not path.exists():path=c.OLD/f'retarget-cnn/seed{s}/fold{k}-{stage}'
  else:path=c.HERE/f'cnn-dense-{kind}/seed{s}/fold{k}-{stage}'
  z=np.load(_repo_io(path/'predictions.npz'));pp.append(z['validation_prediction']);q=z['validation_index']
 return q,np.mean(pp,0)

def main(which):
 d=c.inputs();base=np.load(_repo_io(c.OLD/'baseline076.npz'));reference=np.load(_repo_io(c.EXP/'highnoise-relational-20261016/inputs.npz'));p82=reference['base082'];y=d['y'][base['dev']];hi=(base['noise']>=12)&~base['failed'];ff=base['fold'];assert np.array_equal(reference['dev'],base['dev']) and abs(rm(p82,y)-9.256738746625928)<1e-10;fixed=json.loads((c.R4/'noise-calibration/summary.json').read_text())['aux']['selection'];pos={int(q):i for i,q in enumerate(base['dev'])};pred={};meta=[]
 if which in ['retarget','dense']:
  assert json.loads((c.HERE/('cnn-retarget-complete.json' if which=='retarget' else 'dense-complete.json')).read_text())['complete'];arms=['old','retarget'] if which=='retarget' else ['old','pick','pool']
  for arm in arms:
   for suffix in ['matched','frozen','seed0','seed1']:pred[arm+'_'+suffix]=base['p76'].copy()
  for k in range(5):
   t=np.load(_repo_io(c.teacher_path(k,'outer0')));anchor=np.load(_repo_io(c.R4/f'anchors/fold{k}.npz'));B=t['query'];assert np.array_equal(B,anchor['B']);good=np.nan_to_num(d['X'][B,list(d['keys']).index('n_obj')],nan=0)>=20;pb=.16*anchor['prediction_B']+.84*t['base'];pb[~good]=anchor['prediction_B'][~good];band=d['noise'][B][good]>=12;record=dict(fold=k,selection={})
   for arm in arms:
    for seed in [None,0,1]:
     qb,rb=corrections(k,'selector',arm,seed);V,rv=corrections(k,'final',arm,seed);assert np.array_equal(qb,B) and set(B).isdisjoint(V);w,mm=cr.select(rb[good],(d['y'][B]-pb)[good],band);altered=d['y'].copy();altered[V]=987654.;ww,m2=cr.select(rb[good],(altered[B]-pb)[good],band);assert np.array_equal(w,ww) and mm==m2;ix=np.array([pos[int(v)] for v in V]);act=hi[ix];name=arm+('_matched' if seed is None else '_seed'+str(seed));pred[name][ix[act]]+=w[1]*rv[act];record['selection'][name]=dict(weights=w.tolist(),**mm,outer_label_poison_exact=True)
     if seed is None:pred[arm+'_frozen'][ix[act]]+=fixed[k]['addition']['weights'][1]*rv[act]
   meta.append(record)
  assert np.max(abs(pred['old_frozen']-p82))<1e-10
 else:
  z=np.load(_repo_io(c.HERE/'log-predictions.npz'));assert np.array_equal(z['ids'],base['ids']);pred={a:p82.copy() for a in ['smear','offset']}
  for a in pred:pred[a][hi]+=.128*(z[a][hi]-base['logridge076'][hi])
 results={a:compare(p82,p,y,ff,hi) for a,p in pred.items()};primary='retarget_matched' if which=='retarget' else 'pool_matched' if which=='dense' else 'smear';control='old_matched' if which=='retarget' else 'pick_matched' if which=='dense' else 'offset';paired=compare(pred[control],pred[primary],y,ff,hi);m=results[primary];advance=m['overall_gain']>=.02 and m['high_gain']>0 and m['improved_folds']>=3 and m['family_ci'][0]>0 and m['top10_removed_gain']>0 and paired['overall_gain']>0 and (paired['family_ci'][0]>0 if which!='log' else True)
 c.arrays(which+'-oof.npz',ids=base['ids'],dev=base['dev'],y=y,fold=ff,high=hi,base082=p82,base076=base['p76'],**pred)
 with (c.HERE/(which+'-per-image.csv')).open('x') as f:
  w=csv.writer(f);w.writerow(['ID','fold','high','hardness','082',*pred,'primary_SSE_gain'])
  for j in range(len(y)):w.writerow([base['ids'][j],int(ff[j]),bool(hi[j]),y[j],p82[j],*[pr[j] for pr in pred.values()],(p82[j]-y[j])**2-(pred[primary][j]-y[j])**2])
 out=dict(protocol=which,baseline082=dict(overall=rm(p82,y),high=rm(p82[hi],y[hi])),results=results,primary=primary,control=control,primary_vs_control=paired,selections=meta,advance=bool(advance),code_sha256=c.sha(__file__),plan_sha256=c.sha(c.HERE/'PLAN.md'),limitations='Reused400 labels; inherited frozen representations and historical selection; conditional bootstrap, not independent generalization evidence.')
 if which=='retarget':
  pilot=ff<2;tail=ff>=2;out['previous_pilot_vs_completion']={n:dict(n=int(ix.sum()),gain_vs082=rm(p82[ix],y[ix])-rm(pred[primary][ix],y[ix]),gain_vs_matched=rm(pred[control][ix],y[ix])-rm(pred[primary][ix],y[ix])) for n,ix in [('prior2fold',pilot),('remaining3fold',tail)]}
 c.save(_repo_io(which+'-results.json'),out);print(json.dumps({k:v for k,v in out.items() if k not in ['selections']},indent=2),flush=True)
if __name__=='__main__':sys.addaudithook(c.guard);main(sys.argv[1])
