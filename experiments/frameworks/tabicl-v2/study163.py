"""Study163: pre-fixed eight-estimator extension; reuse audited fold engine."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study162 as s
import sys,json,time,hashlib
import numpy as np
BASE=s.P
P=BASE/'study163'
s.P=P
s.PARAMS={**s.PARAMS,'n_estimators':8,'norm_methods':['none','power'],'feat_shuffle_method':'latin','outlier_threshold':4.0}
SPLITS=['original','20261031','20261101']
def summary(split,arm):
 ids,C,B,y,folds,ref,sources=s.inputs(split);pred=np.full(400,np.nan)
 for f in sorted(set(folds)):
  d=np.load(_repo_io(P/'results'/split/arm/str(f)/'prediction.npz'),allow_pickle=False);v=np.where(folds==f)[0]
  assert np.array_equal(ids[v],d['ids']);pred[v]=d['prediction']
 assert np.isfinite(pred).all();blend=.8*ref+.2*pred
 old=np.load(_repo_io(BASE/'results/original'/arm/'oof.npz'),allow_pickle=False) if split=='original' else None
 if old is not None: assert np.array_equal(old['ids'],ids)
 gains=[s.rmse(y[folds==f],ref[folds==f])-s.rmse(y[folds==f],blend[folds==f]) for f in sorted(set(folds))]
 out={'standalone':s.rmse(y,pred),'fixed20blend':s.rmse(y,blend),'reference018':s.rmse(y,ref),'fold_blend_gains':gains,'residual_correlation':float(np.corrcoef(y-ref,y-pred)[0,1]),'qualifies_resplits':s.rmse(y,ref)-s.rmse(y,blend)>=.01 and sum(g>0 for g in gains)>=3}
 if old is not None: out['study162']={'standalone':s.rmse(y,old['prediction']),'fixed20blend':s.rmse(y,old['fixed20blend'])}
 dest=P/'results'/split/arm
 np.savez_compressed(_repo_io(dest/'oof.npz'),ids=ids,y=y,fold=folds,reference018=ref,prediction=pred,fixed20blend=blend)
 s.write(dest/'summary.json',out);print('SUMMARY',split,arm,json.dumps(out),flush=True)
 return out

def checked_fold(arm,split,f,benchmark=False):
 r=s.fit_one(arm,split,int(f),benchmark)
 assert r['actual_ensemble_configs']=={'none':4,'power':4},r['actual_ensemble_configs']
 assert r['fit_seconds']+r['predict_seconds']<60,'Stop: per-fold runtime budget exceeded'
 assert r['process_peak_rss_bytes']<4_000_000_000,'Stop: RSS budget exceeded'
 return r

def main():
 mode=sys.argv[1]
 if mode=='plan':
  P.mkdir(exist_ok=False)
  old=json.loads((BASE/'plan.json').read_text());assert old['parameters']['n_estimators']==1
  old_reports=list((BASE/'results').rglob('report.json'));assert len(old_reports)==15
  assert all(json.loads(p.read_text())['actual_ensemble_configs']=={'none':1} for p in old_reports)
  old_files=[p for p in BASE.rglob('*') if p.is_file() and not any(x in p.relative_to(BASE).parts for x in ['.venv','uv-cache','hf-cache','cache','tmp','study163'])]
  keep={str(p.relative_to(s.R)):s.sha(p) for p in old_files}
  keep.update(json.loads((BASE/'preservation-before.json').read_text()))
  s.write(P/'preservation-before.json',keep)
  _,_,_,_,f,_,sources=s.inputs('original')
  plan={'study':163,'previous_study162_n_estimators':1,'parameters':s.PARAMS,'arms':s.ARMS,'preprocessing':'Same as162. Compact26 raw; DINO-B6912: StandardScaler fit on outer train then full-SVD PCA64 fit on outer train; concatenate these for90. TabICL internal normalizers fitted on train only. No synthetic sample augmentation. Eight deterministic ensemble transformations from seed42, Latin feature permutation, none/power normalization.','benchmark':'original compact26 fold0; CPU n_jobs2. fit+predict<60s and process peak RSS<4e9 bytes required before continuation; same limit checked after each fold. batch_size1 unchanged.','fixed_blend':'0.8 reference018 + 0.2 TabICL; no fitted blend weights','resplit_gate':'original fixed20blend improves RMSE by at least0.01 and >=3/5 folds; qualifying arms only','continuation_gate':'improvement at least0.01 on each of all3 split RMSE plus >=3/5 fold gains per split; otherwise no full500/test. Any learned combination requires separate honest nesting.','sources':sources,'frozen_engine_sha256':s.sha(BASE/'study162.py'),'wrapper_sha256':s.sha(BASE/'study163.py'),'environment':'Existing isolated .venv and checkpoint; no installation or downloads','excluded':'holdout100 and test untouched; no new raw image access','limitations':'same adaptive400 development IDs; resplits not independent data; benchmark fold has seen prior model selection'}
  s.write(P/'plan.json',plan);print(json.dumps(plan,indent=2))
 elif mode=='benchmark':
  checked_fold('compact26','original',0,True)
 elif mode=='original':
  assert (P/'results/original/compact26/0/report.json').exists()
  for arm in s.ARMS:
   for f in sorted(set(s.inputs('original')[4])): checked_fold(arm,'original',f)
   summary('original',arm)
 elif mode=='resplits':
  arms=[a for a in s.ARMS if json.loads((P/'results/original'/a/'summary.json').read_text())['qualifies_resplits']]
  for split in SPLITS[1:]:
   for arm in arms:
    for f in sorted(set(s.inputs(split)[4])):checked_fold(arm,split,f)
    summary(split,arm)
  print('RESPLIT_ARMS',arms,flush=True)
 else:raise ValueError(mode)
if __name__=='__main__':main()
