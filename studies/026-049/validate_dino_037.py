"""Independent artifact, leakage-boundary and portable model reload checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(ROOT))
import numpy as np
import torch,joblib
import dino_regression_037
import dino_frozen_037 as df

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
root=_repo_path(ROOT, 'artifacts/dino-frozen-037');out=_repo_path(root, 'regression');plan=json.loads((_repo_path(out, 'plan.json')).read_text());selection=json.loads((_repo_path(out, 'selection.json')).read_text());splits=['original']+(['20261031','20261101'] if selection['selected'] else [])
for n,h in {**plan['source_sha256'],**plan['protected']}.items():assert sha(_repo_path(ROOT, n))==h
assert sha(_repo_path(root, 'development_features.npz'))==plan['feature_sha256']
with np.load(_repo_path(root, 'development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();X=z['X'].astype(np.float64)
with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert ids==z['ids'].tolist();y=z['y'].copy()
assert len(ids)==len(set(ids))==400 and X.shape==(400,1152) and np.isfinite(X).all()
assert float(X.std(axis=0).min())>0
smoke=json.loads((_repo_path(root, 'smoke.json')).read_text());extract=json.loads((_repo_path(root, 'extraction.json')).read_text())
assert extract['weights_unchanged'] and smoke['frozen_eval_and_no_grad']
for n,h in smoke['official_source_sha256'].items():assert sha(df.SOURCE/n)==h
for n,h in smoke['metadata_sha256'].items():assert sha(df.SOURCE/n)==h
for i,h in extract['image_sha256'].items():assert i in ids and sha(_repo_path(ROOT, 'data/train', f'{i}.png'))==h
# CPU fresh-load reproduction of first4 development rows; no target-based image choice.
torch.set_num_threads(2);model=df.model();input=torch.from_numpy(np.stack([df.preprocess(i) for i in ids[:4]]));reproduced=df.descriptor(model,input).numpy();np.testing.assert_allclose(reproduced,X[:4],atol=2e-3,rtol=2e-3);feature_delta=float(abs(reproduced-X[:4]).max());del model
results={};nmodels=0;maxreload=0.;maxref=0.
for split in splits:
 d=_repo_path(out, split);r=json.loads((d/'report.json').read_text())
 with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
 assert [a['ID'] for a in rows]==ids;np.testing.assert_array_equal(y,[float(a['hardness']) for a in rows]);p={n:np.array([float(a[n]) for a in rows]) for n in r['scores']}
 for n,a in p.items():assert np.isfinite(a).all() and abs(rmse(y,a)-r['scores'][n]['rmse'])<1e-10
 arms=list(r['folds'][0]['models'])
 for a in arms:np.testing.assert_allclose(p[a+'_blend'],.75*p['fresh003']+.25*p[a],rtol=0,atol=1e-12)
 for k,f in enumerate(r['folds']):
  tr=f['train_ids'];va=f['validation_ids'];assert len(tr)==320 and len(va)==80 and not set(tr)&set(va) and set(tr)|set(va)==set(ids);ti=np.array([ids.index(i) for i in tr]);vi=np.array([ids.index(i) for i in va]);assert all(int(rows[j]['fold'])==k for j in vi)
  fd=d/f'fold-{k}'
  for n,h in f['sha256'].items():assert sha(fd/n)==h
  cp=torch.load(_repo_io(fd/'tiny128.pt'),map_location='cpu',weights_only=True);assert cp['train_ids']==tr and cp['validation_ids']==va
  maxref=max(maxref,f['reference003_max_abs_drift'])
  for a in arms:
   obj=joblib.load(_repo_io(fd/f'{a}.joblib'));assert obj['train_ids']==tr and obj['validation_ids']==va;m=obj['model'];assert type(m).__module__=='dino_regression_037';np.testing.assert_allclose(m.scaler.mean_,m.select(X[ti]).mean(0),atol=1e-10,rtol=0);assert int(m.scaler.n_samples_seen_)==320
   assert abs(m.ymean-y[ti].mean())<1e-10 and abs(m.ystd-y[ti].std())<1e-10
   if m.pca is not None:
    assert m.pca.n_components_==64 and m.pca.n_samples_==320;np.testing.assert_allclose(m.pca.mean_,m.scaler.transform(m.select(X[ti])).mean(0),atol=1e-10,rtol=0)
   predicted=m.predict(X[vi]);delta=float(abs(predicted-p[a][vi]).max());maxreload=max(maxreload,delta);np.testing.assert_allclose(predicted,p[a][vi],rtol=0,atol=1e-9)
   rec=f['models'][a];assert m.alpha==min(rec['inner_scores'],key=lambda s:s['inner_rmse'])['alpha']
   for it,iv in rec['inner_train_ids_indices']:assert not set(it)&set(iv) and set(it)|set(iv)==set(ti) and not (set(it)|set(iv))&set(vi)
   nmodels+=1
 results[split]={'scores':r['scores'],'seconds':r['elapsed_seconds'],'chosen_alphas':{a:[f['models'][a]['chosen_alpha'] for f in r['folds']] for a in arms},'mean_train_rmse':{a:float(np.mean([f['models'][a]['train_rmse'] for f in r['folds']])) for a in arms},'oof_sha256':sha(d/'development_oof.csv')}
scores=results['original']['scores'];eligible=[a for a in plan['arms'] if scores[a+'_blend']['comparison']['gain']>0 and sum(g>0 for g in scores[a+'_blend']['comparison']['fold_gains'])>=3];chosen=min(eligible,key=lambda a:scores[a+'_blend']['rmse']) if eligible else None;assert selection['selected']==chosen and selection['eligible']==eligible
summary={'validation':{'development_rows':400,'feature_width':1152,'separate_process_regressor_reloads':nmodels,'max_regressor_reload_error':maxreload,'max_reference003_drift':maxref,'cpu_reload_vs_mps_features_max_abs_difference':feature_delta,'all_id_finite_rmse_fixedblend_checks':True,'fold_only_scaler_pca_target_checks':True,'inner_split_disjoint_outer_validation':True,'screening_gate_reproduced':True,'source_license_weights_and_artifact_hash_checks':True,'best003_sha256':sha(_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'))},'results':results,'selected':chosen,'extraction_seconds':extract['seconds'],'total_cv_seconds':sum(a['seconds'] for a in results.values()),'limits':['Historical development400 reuse and original3arm screening produce selection optimism; repartitions are not independent unseen validation.','Conditional bootstrap ignores fitting and model-selection uncertainty.','252fullimage resize is a predetermined task-specific choice, not official224center-crop evaluation; only ImageNetnormalization matches officialdefaults.','No holdout/test access or backbone adaptation; cached features are image-local and frozen.']}
path=_repo_path(ROOT, 'reports/dino-037-validation.json')
if path.exists():assert json.loads(path.read_text())==summary
else:path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
