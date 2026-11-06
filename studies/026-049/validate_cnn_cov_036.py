"""Verify036 artifacts and independently reproduce the fixed screening decision."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import sys,csv,json,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
import numpy as np
import torch,joblib

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
root=_repo_path(ROOT, 'artifacts/cnn-cov-controls-036');plan=json.loads((_repo_path(root, 'plan.json')).read_text());sel=json.loads((_repo_path(root, 'selection.json')).read_text());splits=['original']+(['20261031','20261101'] if sel['selected'] else [])
for n,h in {**plan['source_sha256'],**plan['protected']}.items():assert sha(_repo_path(ROOT, n))==h
with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();y=z['y'].copy();X=z['X'].copy()
assert len(ids)==len(set(ids))==400
results={};count=0;drift=0.;reload_max=0.
for split in splits:
 d=_repo_path(root, split);r=json.loads((d/'report.json').read_text())
 with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
 assert [a['ID'] for a in rows]==ids;np.testing.assert_array_equal(y,[float(a['hardness']) for a in rows])
 p={n:np.array([float(a[n]) for a in rows]) for n in r['scores']}
 for n,a in p.items():assert np.isfinite(a).all() and abs(rmse(y,a)-r['scores'][n]['rmse'])<1e-10
 with (_repo_path(ROOT, 'artifacts/cnn-structure-035', split, 'development_oof.csv')).open() as f:previous=list(csv.DictReader(f))
 assert [a['ID'] for a in previous]==ids
 actual=np.array([float(a['recorded003']) for a in previous]);drift=max(drift,float(abs(actual-p['reference003']).max()));np.testing.assert_allclose(actual,p['reference003'],atol=1e-10,rtol=0)
 names=list(r['folds'][0]['models']);diagnostics={}
 for name in names:
  np.testing.assert_allclose(p[name+'_blend'],.5*p['baseline']+.25*p['compact_kernel']+.25*p[name],atol=1e-12,rtol=0)
  a=[f['models'][name] for f in r['folds']]
  diagnostics[name]={'mean_train_rmse':float(np.mean([x['train_rmse'] for x in a])),'mean_validation_rmse':float(np.mean([x['validation_rmse'] for x in a])),'validation_epoch60_to80_change':float(np.mean([x['curve'][79]['validation_rmse_diagnostic']-x['curve'][59]['validation_rmse_diagnostic'] for x in a])),'baseline_residual_correlation':float(np.corrcoef(y-p['baseline'],y-p[name])[0,1])}
 for k,f in enumerate(r['folds']):
  tr=f['train_ids'];va=f['validation_ids'];assert len(tr)==320 and len(va)==80 and not set(tr)&set(va) and set(tr)|set(va)==set(ids)
  ti=np.array([ids.index(i) for i in tr]);assert all(int(rows[ids.index(i)]['fold'])==k for i in va)
  for n,h in f['checkpoint_sha256'].items():assert sha(d/f'fold-{k}'/n)==h
  for name in names:
   cp=torch.load(_repo_io(d/f'fold-{k}'/f'{name}.pt'),map_location='cpu',weights_only=True);assert cp['train_ids']==tr and cp['validation_ids']==va and cp['epochs']==80 and cp['seed']==20261003+k and cp['image_size']==128
   assert abs(cp['normalization']['y_mean']-y[ti].mean())<1e-10 and abs(cp['normalization']['y_std']-y[ti].std())<1e-10
   sc=joblib.load(_repo_io(d/f'fold-{k}'/f'{name}-morph-scaler.joblib'));mm,ss=sc.steps[0][1],sc.steps[1][1]
   np.testing.assert_allclose(mm.data_min_,X[ti].min(0),atol=1e-12,rtol=0);np.testing.assert_allclose(mm.data_max_,X[ti].max(0),atol=1e-12,rtol=0);np.testing.assert_allclose(ss.mean_,mm.transform(X[ti]).mean(0),atol=1e-12,rtol=0)
   reload_max=max(reload_max,f['models'][name]['reload_max_abs_difference']);count+=1
 results[split]={'scores':r['scores'],'diagnostics':diagnostics,'seconds':r['elapsed_seconds'],'report_sha256':sha(d/'report.json'),'oof_sha256':sha(d/'development_oof.csv')}
scores=results['original']['scores'];arms=['small_trace','small_regularized'];eligible=[n for n in arms if scores[n+'_blend']['comparison']['gain']>0 and sum(x>0 for x in scores[n+'_blend']['comparison']['fold_gains'])>=3];chosen=min(eligible,key=lambda n:scores[n+'_blend']['rmse']) if eligible else None;assert chosen==sel['selected'] and eligible==sel['eligible']
summary={'validation':{'checkpoints':count,'partitions':splits,'rows_per_partition':400,'oof_ids_finite_rmse_blend_checks':True,'fold_only_preprocessing_and_disjoint_ids':True,'source_and_checkpoint_hashes':True,'fixed_screening_gate_reproduced':True,'mps_reload_max_abs_diff':reload_max,'reference003_vs_actual003_max_abs_diff':drift,'best003_sha256':sha(_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'))},'selected_for_confirmation':chosen,'results':results,'training_seconds':sum(a['seconds'] for a in results.values()),'reporting_annotation_correction':'036 field fixed_035_controls_reused_for_reporting_only means no control predictions enter model training inputs/targets; they ARE used as fixed comparison and screening selection references. Actual recorded003 identity is verified here.','limitations':['Historical reuse of development400 creates repeated-selection bias. The repartitions are not independent unseen data.','Screening chooses one of two arms and adds selection optimism. Conditional bootstrap intervals omit model fitting and selection uncertainty.','The two new arms have identical initialization; comparisons against035 differ in capacity and initialization consumption.','No holdout/test use or external data/weights. No public score used in model selection.']}
path=_repo_path(ROOT, 'reports/cnn-cov-controls-036-validation.json')
if path.exists():assert json.loads(path.read_text())==summary
else:path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
