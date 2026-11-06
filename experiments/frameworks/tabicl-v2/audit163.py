"""Independent reload/lineage audit; no network, holdout or test access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study163 as w
s=w.s;P=w.P;BASE=w.BASE;R=s.R
import numpy as np,json,time,socket,gc
start=time.perf_counter()
def deny(*args,**kwargs):raise RuntimeError('Network disabled for study163 audit')
socket.socket.connect=deny;socket.create_connection=deny
rows=[]
for report_path in sorted((P/'results').glob('*/*/*/report.json')):
 r=json.loads(report_path.read_text());split,arm,fold=r['split'],r['arm'],r['fold'];out=report_path.parent
 ids,C,B,y,folds,ref,sources=s.inputs(split);t=np.where(folds!=fold)[0];v=np.where(folds==fold)[0]
 assert r['train_ids']==ids[t].tolist() and r['validation_ids']==ids[v].tolist() and not set(ids[t])&set(ids[v])
 assert s.sha(out/'prediction.npz')==r['prediction_sha256'] and s.sha(out/'model.pkl')==r['model_sha256']
 if arm=='compact26':xt,xv=C[t],C[v]
 else:
  tr=s.joblib.load(_repo_io(out/'transform.joblib'));ss,pca=tr['scaler'],tr['pca']
  np.testing.assert_allclose(ss.mean_,B[t].mean(axis=0,dtype=np.float64),rtol=0,atol=1e-12)
  assert ss.n_samples_seen_==320 and pca.n_samples_==320
  xt=pca.transform(ss.transform(B[t]));xv=pca.transform(ss.transform(B[v]))
  if arm=='compact26_dinob_pca64':xt=np.column_stack([C[t],xt]);xv=np.column_stack([C[v],xv])
 model=s.TabICLRegressor.load(out/'model.pkl',device='cpu')
 assert model.n_estimators==8 and model.allow_auto_download is False and model.random_state==42
 assert model.device_==s.torch.device('cpu') and model.n_samples_in_==320
 np.testing.assert_allclose(model.y_scaler_.mean_,[y[t].astype(np.float32).mean(dtype=np.float64)],rtol=0,atol=1e-12)
 eg=model.ensemble_generator_;assert {k:len(v) for k,v in eg.ensemble_configs_.items()}=={'none':4,'power':4}
 xt_encoded=eg.unique_filter_.transform(model.X_encoder_.transform(xt))
 for pp in eg.preprocessors_.values():np.testing.assert_allclose(pp.standard_scaler_.mean_,xt_encoded.mean(axis=0),rtol=1e-6,atol=1e-6)
 saved=np.load(_repo_io(out/'prediction.npz'),allow_pickle=False);assert np.array_equal(saved['ids'],ids[v])
 pred=model.predict(xv);assert np.isfinite(pred).all();delta=float(np.max(np.abs(pred-saved['prediction'])));assert delta==0
 row={'split':split,'arm':arm,'fold':fold,'reload_max_abs':delta,'train_only_target_and_feature_scalers_verified':True,'pca_train_only_verified':arm!='compact26','configs':r['actual_ensemble_configs']}
 if split=='original' and fold==0:
  poisoned=y.copy();poisoned[v]=np.arange(len(v))*12345.+1e8
  xx,xq,_=s.features(arm,C,B,t,v);fresh=s.TabICLRegressor(**s.PARAMS).fit(xx,poisoned[t])
  delta=float(np.max(np.abs(fresh.predict(xq)-pred)));assert delta==0
  row['validation_label_poison_fresh_fit_max_abs']=delta;del fresh
 rows.append(row);del model;gc.collect()
 print('AUDITED',split,arm,fold,flush=True)
old=json.loads((P/'preservation-before.json').read_text())
# Shell redirection created this new plan stdout file just before snapshot; it was then populated by this study.
excluded={'experiments/tabicl-v2/study163-plan.log':'New study163 stdout file was open during snapshot, not a pre-existing study162 artifact.'}
checked={k:v for k,v in old.items() if k not in excluded}
assert all(s.sha(R/k)==v for k,v in checked.items())
assert all(s.sha(R/k)==v for k,v in json.loads((BASE/'preservation-before.json').read_text()).items())
plan=json.loads((P/'plan.json').read_text());assert s.sha(BASE/'study162.py')==plan['frozen_engine_sha256'];assert s.sha(BASE/'study163.py')==plan['wrapper_sha256']
audit={'status':'passed','offline_socket_connections_blocked':True,'models_reloaded':len(rows),'replays':rows,'preserved_files':checked,'snapshot_new_log_exclusion':excluded,'plan_sha256':s.sha(P/'plan.json'),'audit_script_sha256':s.sha(BASE/'audit163.py'),'seconds':time.perf_counter()-start}
s.write(P/'audit.json',audit);print('AUDIT_PASS',len(rows),time.perf_counter()-start,flush=True)
