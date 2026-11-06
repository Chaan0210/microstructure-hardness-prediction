"""Only score locked complete5-fold packages after independent CPU replay passes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
O=Path(__file__).resolve().parent;S=O.parent;R=_RepoPath(__file__).resolve().parents[4];sys.path.insert(0,str(S))
import numpy as np
from contracts import sha,require,lock_manifest
from evaluation import evaluate_all

def main():
 start=time.perf_counter();complete=json.loads((_repo_path(O, 'fit-completion.json')).read_text());require(complete['passed'],'Main incomplete');independent_path=_repo_path(S, 'independent-check/results.json');independent=json.loads(independent_path.read_text());require(independent['passed'],'Independent validation incomplete')
 contract=json.loads((_repo_path(S, 'data-contract.json')).read_text());require(sha(_repo_path(O, 'oof-unscored.npz'))==complete['oof_sha256'],'OOF hash');seen=[]
 for k in range(5):
  folder=_repo_path(O, f'fold-{k}');require(sha(folder/'validation.json')==complete['fold_validation_sha256'][str(k)],'Fold verification changed');v=json.loads((folder/'validation.json').read_text());require(v['passed'] and v['V']==contract['graphs'][k]['V'],'Incomplete/changed validation fold');seen.extend(v['V'])
  for name,key in [('frozen-anchor.joblib','anchor_sha256'),('predictions.npz','predictions_sha256'),('training-report.json','training_report_sha256')]:require(sha(folder/name)==v[key],'Fold source mismatch')
  for arm,h in v['bundles'].items():
   require(sha(folder/arm/'manifest.json')==h,'Bundle manifest mismatch');b=json.loads((folder/arm/'manifest.json').read_text());require(sha(folder/arm/'anchor.joblib')==b['anchor_sha256']==v['anchor_sha256'],'Bundle anchor changed');require(sha(folder/arm/'residual.joblib')==b['residual_sha256'],'Bundle residual changed');g=contract['graphs'][k];require(b['anchor_train']==g['A'] and b['residual_train']==g['B'],'Bundle split mismatch');require(b['anchor_ids']==[contract['ids'][i] for i in g['A']] and b['residual_ids']==[contract['ids'][i] for i in g['B']],'Bundle ID mismatch')
 require(sorted(seen)==list(range(400)),'Validation union not400 unique')
 with np.load(_repo_path(O, 'oof-unscored.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 require(d['ids'].tolist()==contract['ids'],'OOF IDs');source=_repo_path(R, 'experiments/cnn-nested-204/cache/data.npz');require(sha(source)==contract['inputs']['experiments/cnn-nested-204/cache/data.npz'],'204 cache changed')
 with np.load(_repo_io(source),allow_pickle=False) as z:require(z['ids'].tolist()==d['ids'].tolist(),'Incumbent IDs');np.testing.assert_array_equal(z['fold'][0],d['fold']);y=z['y'].copy();incumbent=z['R'][0].copy();groups=z['groups'].copy()
 p204=json.loads((_repo_path(R, 'experiments/cnn-nested-204/plan.json')).read_text());ref='experiments/dinol-177/original/predictions.npz';require(sha(_repo_path(R, ref))==p204['inputs_sha256'][ref],'Original best021 OOF changed')
 with np.load(_repo_path(R, ref),allow_pickle=False) as z:
  np.testing.assert_array_equal(z['ids'],d['ids']);np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['g'],groups);np.testing.assert_array_equal(z['fold'],d['fold']);np.testing.assert_allclose(z['blend_L_MLP'],incumbent,atol=1e-9,rtol=0)
 result=evaluate_all(y,d['R0'],d['R1'],d['Rperm'],incumbent,np.asarray(d['fold']),list(range(5)))
 # Descriptive only, thresholds never used for selection, weighting or trimming.
 thresholds=np.quantile(y,[.1,.9]);masks={'lower_target_decile':y<=thresholds[0],'middle_target80pct':(y>thresholds[0])&(y<thresholds[1]),'upper_target_decile':y>=thresholds[1],'known_three':np.isin(d['ids'],['TRAIN_000206','TRAIN_000223','TRAIN_000482'])};desc={}
 for name,mask in masks.items():
  desc[name]={'n':int(mask.sum()),'RMSE':{key:float(np.sqrt(np.mean((y[mask]-p[mask])**2))) for key,p in [('R0',d['R0']),('R1',d['R1']),('Rperm',d['Rperm']),('P021',incumbent)]}}
 result.update(passed=True,descriptive_only=desc,descriptive_target_thresholds=thresholds.tolist(),fit_completion_sha256=sha(_repo_path(O, 'fit-completion.json')),independent_validation_sha256=sha(independent_path),incumbent_original177_sha256=sha(_repo_path(R, ref)),incumbent204_sha256=sha(source),OOF_sha256=sha(_repo_path(O, 'oof-unscored.npz')),all_five_bundles_hash_and_ID_checked=True,seconds=time.perf_counter()-start,no_new_model_fit=True)
 lock_manifest(_repo_path(O, 'scores.json'),result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
