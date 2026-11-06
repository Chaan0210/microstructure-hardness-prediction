"""Read-only validation of frozen development study; writes a new audit report."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import csv, hashlib, json, sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
import numpy as np
import torch, joblib

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rmse(y,p): return float(np.sqrt(np.mean((y-p)**2)))
root=_repo_path(ROOT, 'artifacts/cnn-structure-035')
bench=json.loads((_repo_path(root, 'benchmark.json')).read_text())
for n,h in bench['source_sha256'].items(): assert sha(_repo_path(ROOT, n))==h
with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:
 ids=z['ids'].tolist(); X=z['X'].copy(); y=z['y'].copy()
assert len(ids)==len(set(ids))==400
results={}; checked_checkpoints=0; checkpoint_reload_max=0.
for split in ('original','20261031','20261101'):
 d=_repo_path(root, split); r=json.loads((d/'report.json').read_text())
 with (d/'development_oof.csv').open() as f: rows=list(csv.DictReader(f))
 assert len(rows)==400 and [x['ID'] for x in rows]==ids
 np.testing.assert_array_equal(y,[float(x['hardness']) for x in rows])
 p={n:np.array([float(x[n]) for x in rows]) for n in r['scores']}
 for n,a in p.items():
  assert np.isfinite(a).all()
  assert abs(rmse(y,a)-r['scores'][n]['rmse'])<1e-10
 for n in ('tiny128','cov128','covmorph128','tiny256'):
  np.testing.assert_allclose(p[n+'_blend'],.5*p['baseline']+.25*p['compact_kernel']+.25*p[n],atol=1e-12,rtol=0)
 np.testing.assert_allclose(p['tiny128_blend'],p['recorded003'],atol=2e-4,rtol=0)
 for n,h in r['plan']['protected'].items(): assert sha(_repo_path(ROOT, n))==h
 for k,f in enumerate(r['folds']):
  tr=f['train_ids']; va=f['validation_ids']; assert len(tr)==320 and len(va)==80
  assert not set(tr)&set(va) and set(tr)|set(va)==set(ids)
  ti=np.array([ids.index(i) for i in tr]); vi=np.array([ids.index(i) for i in va])
  assert all(int(rows[j]['fold'])==k for j in vi)
  fd=d/f'fold-{k}'
  for n,h in f['checkpoint_sha256'].items(): assert sha(fd/n)==h
  for name in ('tiny128','cov128','covmorph128','tiny256'):
   cp=torch.load(_repo_io(fd/f'{name}.pt'),map_location='cpu',weights_only=True)
   assert cp['train_ids']==tr and cp['validation_ids']==va and cp['epochs']==80
   assert cp['seed']==20261003+k
   assert abs(cp['normalization']['y_mean']-float(y[ti].mean()))<1e-10
   assert abs(cp['normalization']['y_std']-float(y[ti].std()))<1e-10
   scale=joblib.load(_repo_io(fd/f'{name}-morph-scaler.joblib')); mm,ss=scale.steps[0][1],scale.steps[1][1]
   np.testing.assert_allclose(mm.data_min_,X[ti].min(0),atol=1e-12,rtol=0)
   np.testing.assert_allclose(mm.data_max_,X[ti].max(0),atol=1e-12,rtol=0)
   np.testing.assert_allclose(ss.mean_,mm.transform(X[ti]).mean(0),atol=1e-12,rtol=0)
   assert np.isfinite(scale.transform(X)).all()
   checkpoint_reload_max=max(checkpoint_reload_max,f['models'][name]['reload_max_abs_difference'])
   checked_checkpoints+=1
 scores={n:r['scores'][n] for n in r['scores']}
 curves={}
 for name in ('tiny128','cov128','covmorph128','tiny256'):
  q=[f['models'][name] for f in r['folds']]
  curves[name]={'mean_train_rmse':float(np.mean([a['train_rmse'] for a in q])), 'mean_validation_rmse':float(np.mean([a['validation_rmse'] for a in q])), 'mean_validation_change_epoch60_to80':float(np.mean([a['curve'][79]['validation_rmse_diagnostic']-a['curve'][59]['validation_rmse_diagnostic'] for a in q])), 'train_seconds':float(sum(a['seconds'] for a in q)), 'baseline_residual_correlation':float(np.corrcoef(y-p['baseline'],y-p[name])[0,1])}
 results[split]={'scores':scores,'learning_diagnostics':curves,'elapsed_seconds':r['elapsed_seconds'],'report_sha256':sha(d/'report.json'),'oof_sha256':sha(d/'development_oof.csv')}
summary={'study':'cnn-structure-035','validation':{'development_rows_per_partition':400,'partitions':3,'cnn_checkpoints_checked':checked_checkpoints,'all_oof_finite_unique_and_rmse_recomputed':True,'fixed_blend_equations_passed':True,'reference003_reproduced':True,'checkpoint_and_source_hashes_passed':True,'train_validation_id_disjoint':True,'target_and_morphology_train_fold_preprocessing_passed':True,'mps_checkpoint_reload_max_absolute_difference':checkpoint_reload_max,'protected_candidate003_sha256':sha(_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'))},'results':results,'training_total_seconds':sum(x['elapsed_seconds'] for x in results.values()),'limitations':['All partitions reuse the same historically selected development400; this is robustness analysis, not unseen independent validation.','Bootstrap intervals condition on fitted predictions and omit repeated model-selection uncertainty.','Resolution changes relative receptive field and final spatial map as well as retained image detail.','Covariance and morphology arms add a small number of parameters and different random initialization consumption; architecture differences are not entirely paired.','No holdout/test access, full-data fitting or submission in this study.'],'benchmark':bench}
out=_repo_path(ROOT, 'reports/cnn-structure-035-validation.json')
if out.exists():
 assert json.loads(out.read_text())==summary, 'Existing audit differs; preserve both and investigate'
else:
 with out.open('x') as f: json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps({'validation':summary['validation'],'seconds':summary['training_total_seconds'],'scores':{s:{n:x['rmse'] for n,x in a['scores'].items()} for s,a in results.items()}},indent=2))
