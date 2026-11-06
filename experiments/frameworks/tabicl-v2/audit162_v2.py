from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study162 as s
import numpy as np, json, time, importlib.metadata, socket, gc
P=s.P; R=s.R
start=time.perf_counter()
def deny(*args,**kwargs): raise RuntimeError('Network forbidden during offline audit')
socket.socket.connect=deny; socket.create_connection=deny
all_reports=[]; replay=[]
for arm in s.ARMS:
 ids,C,B,y,folds,ref,sources=s.inputs('original')
 preds=np.empty(400)
 for fold in sorted(set(folds)):
  out=P/'results/original'/arm/str(fold);r=json.loads((out/'report.json').read_text())
  t=np.where(folds!=fold)[0];v=np.where(folds==fold)[0]
  assert r['train_ids']==ids[t].tolist() and r['validation_ids']==ids[v].tolist()
  assert not set(r['train_ids'])&set(r['validation_ids'])
  assert s.sha(out/'prediction.npz')==r['prediction_sha256'] and s.sha(out/'model.pkl')==r['model_sha256']
  if arm=='compact26': xt,xv=C[t],C[v]
  else:
   tr=s.joblib.load(_repo_io(out/'transform.joblib')); ss,pca=tr['scaler'],tr['pca']
   np.testing.assert_allclose(ss.mean_,B[t].mean(axis=0,dtype=np.float64),rtol=0,atol=1e-12)
   assert ss.n_samples_seen_==len(t) and pca.n_samples_==len(t)
   xt=pca.transform(ss.transform(B[t]));xv=pca.transform(ss.transform(B[v]))
   if arm=='compact26_dinob_pca64': xt=np.column_stack([C[t],xt]);xv=np.column_stack([C[v],xv])
  model=s.TabICLRegressor.load(out/'model.pkl',device='cpu')
  assert model.allow_auto_download is False and model.device_==s.torch.device('cpu') and model.n_samples_in_==320
  np.testing.assert_allclose(model.y_scaler_.mean_,[y[t].astype(np.float32).mean(dtype=np.float64)],atol=1e-12,rtol=0)
  pp=model.ensemble_generator_.preprocessors_['none']
  train_encoded=model.X_encoder_.transform(xt)
  train_encoded=model.ensemble_generator_.unique_filter_.transform(train_encoded)
  np.testing.assert_allclose(pp.standard_scaler_.mean_,np.mean(train_encoded,axis=0),rtol=1e-6,atol=1e-6)
  saved=np.load(_repo_io(out/'prediction.npz'),allow_pickle=False)['prediction'];p=model.predict(xv)
  assert np.isfinite(p).all();delta=float(np.max(np.abs(saved-p)));assert delta==0
  preds[v]=p
  item={'arm':arm,'fold':int(fold),'reload_max_abs':delta,'train_only_scaler_target_pca_audit':True}
  if fold==0:
   # Entire validation labels are replaced before fresh fit, with no validation access in feature fitting.
   poisoned=y.copy();poisoned[v]=np.arange(len(v))*12345.+1e8
   xx,xq,_=s.features(arm,C,B,t,v)
   fresh=s.TabICLRegressor(**s.PARAMS).fit(xx,poisoned[t]); poison=fresh.predict(xq)
   item['validation_label_poison_fresh_fit_max_abs']=float(np.max(np.abs(saved-poison)))
   assert item['validation_label_poison_fresh_fit_max_abs']==0
   del fresh
  replay.append(item);all_reports.append(r);del model;gc.collect()
 summary=json.loads((P/'results/original'/arm/'summary.json').read_text())
 assert abs(s.rmse(y,preds)-summary['standalone'])<1e-12
 assert not summary['qualifies_resplits']
old=json.loads((P/'preservation-before.json').read_text())
assert all(s.sha(R/k)==v for k,v in old.items())
ck=json.loads((P/'downloads/checkpoint-provenance.json').read_text()); assert s.sha(s.CK)==ck['sha256']
remote=json.loads((P/'downloads/hf-tree.json').read_text());remote=next(x for x in remote if x['path']==s.CK.name)
assert remote['lfs']['oid']==ck['sha256'] and remote['size']==s.CK.stat().st_size
installed={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
license_path=next((P/'.venv/lib/python3.13/site-packages').glob('tabicl-2.2.0.dist-info/licenses/LICENSE'))
license_text=license_path.read_text();assert 'BSD 3-Clause License' in license_text
s.write(P/'installation-verification.json',{'packages':installed,'official_package':'https://pypi.org/project/tabicl/2.2.0/','official_checkpoint':ck,'remote_lfs_sha256_verified':True,'model_license_card':'bsd-3-clause','package_license_file':str(license_path.relative_to(P)),'package_license_sha256':s.sha(license_path),'license_note':'Core code and checkpoint BSD-3-Clause; unused optional forecast code derived from TabPFN-TS has Apache-2.0 notices included in LICENSE. No optional extras installed.','major_version_changes':False,'existing_venv_modified_by_install':False,'authentication_payment_required':False})
s.write(P/'audit.json',{'status':'passed','offline_network_connections_blocked':True,'replays':replay,'all_predictions_finite':True,'models_reloaded':len(replay),'fold0_poison_replays':3,'preserved_preexisting_files':old,'preservation_scope':'Before/after SHA256 of existing environment package METADATA, project root lock files and submissions018/020. Installer exclusively targeted experiments/tabicl-v2/.venv.','elapsed_seconds':time.perf_counter()-start})
result={'study':162,'status':'completed_negative','best_retained':'018','original':{arm:json.loads((P/'results/original'/arm/'summary.json').read_text()) for arm in s.ARMS},'benchmark':json.loads((P/'results/original/compact26/0/report.json').read_text()),'fits':15,'extra_fresh_fit_replays':4,'fold_predict_seconds':[r['predict_seconds'] for r in all_reports],'max_process_rss_bytes':max(r['process_peak_rss_bytes'] for r in all_reports),'resplits_executed':False,'full500_test_csv_executed':False,'reason':'No arm passes preregistered original-fold fixed20blend gate. Do not change ensemble count, preprocessing or blend based on these outcomes.','plan_sha256':s.sha(P/'plan.json'),'implementation_sha256':s.sha(P/'study162.py'),'audit_sha256':s.sha(P/'audit.json'),'limitations':['n_estimators=1 actual normalization none only; result does not exclude gains from separately justified future ensemble studies','Same 400 development IDs reused over many studies; no independent generalization claim','This experiment uses cached image features only; no external training datasets, holdout100 or test read']}
s.write(P/'final.json',result);print(json.dumps({'status':result['status'],'models_reloaded':15,'label_poison_replays':3,'preservation_files':len(old),'seconds':time.perf_counter()-start},indent=2),flush=True)
