"""Confirm frozen connected-boundary features on two existing development partitions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
import frozen_validation as fv

def main():
 out=_repo_path(ROOT, 'artifacts/connected-confirmation-029');out.mkdir(exist_ok=False)
 feature_report=_repo_path(ROOT, 'artifacts/run-028/report.json');cache=_repo_path(ROOT, 'artifacts/run-028/development_features.npz');report=json.loads(feature_report.read_text());names=report['features'];blocks=report['blocks']
 for name in ('connected_grain_features.py','compact_regression.py','frozen_validation.py'):
  assert report['source_sha256'][name]==fv.sha(_repo_path(ROOT, name)), 'Feature provenance source drift: '+name
 assert report['feature_cache_sha256']==fv.sha(cache), 'Feature cache drift'
 sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());ids=sorted(sp['development_folds']);assert len(ids)==400
 with np.load(_repo_io(cache),allow_pickle=False) as z:
  zi=list(z['ids']);assert len(zi)==len(set(zi))==400 and set(zi)==set(ids);X=z['X'][[zi.index(i) for i in ids]].copy()
 assert X.shape==(400,len(names)) and len(names)==len(blocks) and np.isfinite(X).all()
 plan={'scope':'development400 only; two reused partitions, not unseen validation','hypothesis':'Frozen compact26 plus connected-boundary panel provides new stable morphology information','feature_count':len(names),'feature_sha256':fv.sha(cache),'feature_report_sha256':fv.sha(feature_report),'source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ('confirm_connected_029.py','connected_grain_features.py','compact_regression.py','frozen_validation.py')},'split_seeds':[20261031,20261101],'ensemble':{'baseline':.5,'cnn_seed20261003_identity':.25,'connected_kernel':.25},'inner_selection':'same3fold9settingKRR; no outer selection of weights, rounds, or feature thresholds','protected':{n:fv.sha(_repo_path(ROOT, n)) for n in ('artifacts/submission-candidate-003.csv','artifacts/final-model-017/report.json')}}
 fv.write(out/'plan.json',plan);start=time.perf_counter();results=[]
 for seed in plan['split_seeds']:
  parent=_repo_path(ROOT, 'artifacts/frozen-validation-021', f'seed-{seed}');dest=out/f'seed-{seed}';dest.mkdir();split=json.loads((parent/'split_manifest.json').read_text());assert set(split['development_folds'])==set(ids)
  with (parent/'development_oof.csv').open() as f:rr=list(csv.DictReader(f))
  by={r['ID']:r for r in rr};assert len(rr)==len(by)==400 and set(by)==set(ids)
  y=np.array([float(by[i]['hardness']) for i in ids]);fid=np.array([split['development_folds'][i] for i in ids]);groups=np.array([sp['group_by_id'][i] for i in ids]);assert all(int(by[i]['fold'])==fid[j] for j,i in enumerate(ids))
  with (_repo_path(ROOT, 'artifacts/run-028/development_oof.csv')).open() as f:orig=list(csv.DictReader(f))
  origby={r['ID']:r for r in orig};assert len(orig)==len(origby)==400 and set(origby)==set(ids);np.testing.assert_array_equal(y,[float(origby[i]['hardness']) for i in ids])
  baseline=np.array([float(by[i]['baseline']) for i in ids]);cnn=np.array([float(by[i]['cnn_seed20261003_identity']) for i in ids]);old=np.array([float(by[i]['candidate003']) for i in ids]);oldkernel=np.array([float(by[i]['compact_kernel']) for i in ids]);np.testing.assert_allclose(old,.5*baseline+.25*cnn+.25*oldkernel,rtol=0,atol=1e-10)
  pred=np.full(400,np.nan);logs=[]
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);meta=json.loads((parent/f'fold-{k}-report.json').read_text());assert set(meta['train_ids'])=={ids[i] for i in tr} and set(meta['validation_ids'])=={ids[i] for i in va};assert not set(groups[tr])&set(groups[va])
   pred[va],model,detail=fv.kernel_fold(X,y,groups,blocks,tr,va,k);logs.append({'fold':k,**detail});joblib.dump({'model':model,'feature_names':names,'blocks':blocks,'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va]},_repo_io(dest/f'kernel-fold{k}.joblib'))
  new=.5*baseline+.25*cnn+.25*pred;foldgain=[fv.rmse(y[fid==k],old[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)]
  rng=np.random.default_rng(seed);ind=rng.integers(0,400,size=(2000,400));g=np.sqrt(np.mean((y-old)[ind]**2,axis=1))-np.sqrt(np.mean((y-new)[ind]**2,axis=1))
  result={'seed':seed,'candidate003_rmse':fv.rmse(y,old),'connected_kernel_rmse':fv.rmse(y,pred),'connected_candidate_rmse':fv.rmse(y,new),'fold_improvements':foldgain,'better_folds':sum(x>0 for x in foldgain),'conditional_gain_ci95':list(map(float,np.percentile(g,[2.5,97.5]))),'kernel_inner_selection':logs,'parent_oof_sha256':fv.sha(parent/'development_oof.csv'),'parent_manifest_sha256':fv.sha(parent/'split_manifest.json')}
  with (dest/'development_oof.csv').open('x',newline='') as f:
   w=csv.writer(f);w.writerow(['ID','hardness','fold','candidate003','connected_kernel','connected_candidate']);w.writerows([i,y[j],fid[j],old[j],pred[j],new[j]] for j,i in enumerate(ids))
  fv.write(dest/'report.json',result);results.append(result);print(json.dumps(result),flush=True)
 for n,h in plan['source_sha256'].items():assert fv.sha(_repo_path(ROOT, n))==h
 for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
 assert fv.sha(cache)==plan['feature_sha256'] and fv.sha(feature_report)==plan['feature_report_sha256']
 fv.write(out/'report.json',{'plan':plan,'seeds':results,'elapsed_seconds':time.perf_counter()-start,'limits':fv.CAVEAT})
if __name__=='__main__':main()
