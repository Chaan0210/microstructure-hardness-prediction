"""Three predeclared tree controls, nested selection, development400 only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import sys,csv,json,time,hashlib,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor,GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def read(p):return json.loads(p.read_text())
def rows(p,ids,y=None):
 with p.open() as f:r=list(csv.DictReader(f))
 b={z['ID']:z for z in r};assert len(r)==len(b)==400 and set(b)==set(ids)
 if y is not None:np.testing.assert_array_equal(y,[float(b[i]['hardness']) for i in ids])
 return b

def rmse(y,p):
 assert np.isfinite(p).all() and y.shape==p.shape
 return float(np.sqrt(np.mean((y-p)**2)))
def models():
 return {'hgb_old':HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20261024),'hgb_relaxed':HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,min_samples_leaf=5,l2_regularization=1,learning_rate=.05,early_stopping=False,random_state=20261024),'gbr_shallow':GradientBoostingRegressor(n_estimators=300,max_depth=2,min_samples_leaf=10,learning_rate=.03,loss='squared_error',subsample=1.,random_state=20261024)}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--split',default='original',choices=['original','20261031','20261101']);args=parser.parse_args()
 out=_repo_path(ROOT, 'artifacts/run-027', args.split);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
 sp=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));ids=sorted(sp['development_folds']);assert len(ids)==400
 rr=rows(_repo_path(ROOT, 'artifacts/run-012/development_oof.csv'),ids);y=np.array([float(rr[i]['hardness']) for i in ids]);groups=np.array([sp['group_by_id'][i] for i in ids])
 cache=_repo_path(ROOT, 'artifacts/run-022/development_features.npz')
 with np.load(_repo_io(cache),allow_pickle=False) as z:
  zi=list(z['ids']);assert len(set(zi))==400 and set(zi)==set(ids);X=z['X'][[zi.index(i) for i in ids]].copy()
 assert X.shape==(400,57) and np.isfinite(X).all()
 protected={n:sha(_repo_path(ROOT, n)) for n in ('artifacts/submission-candidate-003.csv','artifacts/final-model-017/report.json')}
 if args.split=='original':
  fid=np.array([sp['development_folds'][i] for i in ids]);c=rows(_repo_path(ROOT, 'artifacts/run-018/development_oof.csv'),ids,y)
  baseline=np.array([float(rr[i]['baseline']) for i in ids]);cnn=np.array([float(c[i]['cnn_seed20261003']) for i in ids]);kernel=np.array([float(rr[i]['nested_kernel']) for i in ids]);old=np.array([float(rr[i]['nested_kernel_cnnblend']) for i in ids]);parents=['artifacts/run-012/development_oof.csv','artifacts/run-018/development_oof.csv']
 else:
  p=_repo_path(ROOT, 'artifacts/frozen-validation-021', f'seed-{args.split}');s=read(p/'split_manifest.json');r=rows(p/'development_oof.csv',ids,y);fid=np.array([s['development_folds'][i] for i in ids]);assert all(int(r[i]['fold'])==s['development_folds'][i] for i in ids)
  for k in range(5):
   meta=read(p/f'fold-{k}-report.json');assert set(meta['train_ids'])=={i for j,i in enumerate(ids) if fid[j]!=k};assert set(meta['validation_ids'])=={i for j,i in enumerate(ids) if fid[j]==k}
  baseline=np.array([float(r[i]['baseline']) for i in ids]);cnn=np.array([float(r[i]['cnn_seed20261003_identity']) for i in ids]);kernel=np.array([float(r[i]['compact_kernel']) for i in ids]);old=np.array([float(r[i]['candidate003']) for i in ids]);parents=[str((p/'development_oof.csv').relative_to(ROOT)),str((p/'split_manifest.json').relative_to(ROOT))]
 np.testing.assert_allclose(old,.5*baseline+.25*cnn+.25*kernel,rtol=0,atol=1e-10)
 plan={'hypothesis':'Fixed57 features may be underfit by the old HGB capacity/regularization. Compare one relaxed HGB and one shallow exact-split GBR; no tail weights or prediction expansion.','split':args.split,'models':{n:m.get_params() for n,m in models().items()},'selection':'Inner3 pooled RMSE alone selects among exactly3 profiles; no early stopping. Report all original controls, primary nested replacement and secondary fixed half-kernel half-tree feature blend.','primary_blend':{'baseline':.5,'original_single_cnn':.25,'inner_selected_tree':.25},'secondary_blend':{'baseline':.5,'original_single_cnn':.25,'compact26_kernel':.125,'inner_selected_tree':.125},'inner_seed':20261005,'source_sha256':sha(Path(__file__)),'feature_sha256':sha(cache),'parent_sha256':{p:sha(_repo_path(ROOT, p)) for p in parents},'protected':protected,'scope':'No holdout/test reads, no Public score selection. Cached development features image-local; target unweighted raw. Reused CV is not unseen validation.'}
 write(out/'plan.json',plan)
 preds={n:np.full(400,np.nan) for n in [*models(),'nested_tree']};logs=[]
 for k in range(5):
  tick=time.perf_counter();tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va])
  inner=list(GroupKFold(3,shuffle=True,random_state=20261005+k).split(X[tr],y[tr],groups[tr]));inner_scores={}
  for name in models():
   pp=np.full(len(tr),np.nan)
   for ti,vi in inner:
    m=models()[name].fit(X[tr[ti]],y[tr[ti]]);pp[vi]=m.predict(X[tr[vi]])
   inner_scores[name]=rmse(y[tr],pp)
  chosen=min(inner_scores,key=inner_scores.get);details={}
  for name,m in models().items():
   m.fit(X[tr],y[tr]);preds[name][va]=m.predict(X[va]);curve=[]
   for epoch,(tp,vp) in enumerate(zip(m.staged_predict(X[tr]),m.staged_predict(X[va])),1):
    if epoch in (25,50,100,150,200,250,300):curve.append({'iteration':epoch,'train_rmse':rmse(y[tr],tp),'validation_rmse_diagnostic':rmse(y[va],vp)})
   details[name]={'train_rmse':rmse(y[tr],m.predict(X[tr])),'validation_rmse':rmse(y[va],preds[name][va]),'curve':curve}
   if name==chosen:joblib.dump({'model':m,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'features':read(_repo_path(ROOT, 'artifacts/run-022/report.json'))['features']},_repo_io(out/f'fold{k}-selected.joblib'))
  preds['nested_tree'][va]=preds[chosen][va]
  log={'fold':k,'inner_scores':inner_scores,'chosen':chosen,'models':details,'seconds':time.perf_counter()-tick};logs.append(log);write(out/f'fold{k}.json',log);print(json.dumps({'split':args.split,'fold':k,'chosen':chosen,'scores':{n:d['validation_rmse'] for n,d in details.items()},'seconds':log['seconds']}),flush=True)
 values={'candidate003':old,**preds}
 for n,pred in preds.items():values[n+'_replace']=.5*baseline+.25*cnn+.25*pred
 values['nested_tree_complement']=.5*baseline+.25*cnn+.125*kernel+.125*preds['nested_tree']
 scores={n:{'rmse':rmse(y,p),'fold_rmse':[rmse(y[fid==k],p[fid==k]) for k in range(5)],'residual_correlation_to003':float(np.corrcoef(y-old,y-p)[0,1])} for n,p in values.items()}
 for n in ('nested_tree_replace','nested_tree_complement'):
  rng=np.random.default_rng(20261105);ind=rng.integers(0,400,size=(2000,400));gain=np.sqrt(np.mean((y-old)[ind]**2,axis=1))-np.sqrt(np.mean((y-values[n])[ind]**2,axis=1));scores[n]['conditional_paired_gain_ci95']=list(map(float,np.percentile(gain,[2.5,97.5])))
 with (out/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold',*values]);w.writerows([i,y[j],fid[j],*[v[j] for v in values.values()]] for j,i in enumerate(ids))
 for p,h in protected.items():assert sha(_repo_path(ROOT, p))==h
 report={'plan':plan,'scores':scores,'folds':logs,'elapsed_seconds':time.perf_counter()-started,'limits':'Original400 repeatedly selected. Nested selection controls only3 current tree settings. Repartition uses same400 and is robustness, not unseen validation. Conditional CI ignores fitting and historical selection uncertainty.'};write(out/'report.json',report);print(json.dumps({'split':args.split,'scores':scores,'seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
