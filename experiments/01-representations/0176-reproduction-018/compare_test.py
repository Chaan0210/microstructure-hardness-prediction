"""Compare already-frozen176 inference with018; no fitting or score selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import csv,json,numpy as np
import infer as i
ROOT=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
def diff(a,b):return {'exact':bool(np.array_equal(a,b)),'max_abs':float(np.max(abs(a-b))),'RMS':float(np.sqrt(np.mean((a-b)**2)))}
def main():
 i.guards();v=i.read(_repo_path(ROOT, 'test-run/inference-validation.json'));assert v['passed'];newfile=_repo_path(ROOT, 'test-run/reproduced-018.csv');oldfile=_repo_path(P, 'artifacts/submission-candidate-018.csv');a=list(csv.DictReader(newfile.open()));b=list(csv.DictReader(oldfile.open()));assert [r['ID'] for r in a]==[r['ID'] for r in b];delta=diff(np.array([float(r['hardness']) for r in a]),np.array([float(r['hardness']) for r in b]));stages={};feature_checks={}
 with np.load(_repo_path(ROOT, 'test-run/predictions.npz'),allow_pickle=False) as n,np.load(_repo_path(P, 'artifacts/final-model-145/predictions.npz'),allow_pickle=False) as o:
  np.testing.assert_array_equal(n['ids'],o['ids']);stages['components_without_retired_baseline']=diff(n['P'][:,1:],o['components'][:,1:])
  for key in ['short','base','corrections','primary']:stages[key]=diff(n[key],o[key])
  stages['residual_candidates']=diff(n['residual_candidates'],o['candidate_predictions'])
 with np.load(_repo_path(ROOT, 'test-run/test-features.npz'),allow_pickle=False) as f:
  for name,file in [('S','artifacts/final-model-045/test_features.npz'),('X57','artifacts/final-model-114/test57.npz'),('B','artifacts/final-model-065/test_B_features.npz'),('spatial','artifacts/final-model-141/test-adjacent.npz')]:
   with np.load(_repo_path(P, file),allow_pickle=False) as old:np.testing.assert_array_equal(f['ids'],old['ids']);feature_checks[name]=diff(f[name],old['X'])
 meta=i.read(_repo_path(ROOT, 'test-run/test-feature-validation.json'));assert meta['image_sha256']==i.read(_repo_path(P, 'artifacts/final-model-141/test-adjacent.json'))['image_sha256']
 r={'passed':all(z['exact'] for z in stages.values()) and delta['exact'] and all(z['exact'] for z in feature_checks.values()),'feature_checks':feature_checks,'all_raw_test_image_hashes_match_historical':True,'CSV_bytes_exact':newfile.read_bytes()==oldfile.read_bytes(),'CSV_sha256':i.sha(newfile),'original_CSV_sha256':i.sha(oldfile),'rows':1000,'IDs_order_exact':True,'CSV_difference':delta,'stages':stages,'test_fit_tuning':False,'test_target_values_parsed':False,'fit_backward_guards':True,'original_prediction_sha256':i.sha(_repo_path(P, 'artifacts/final-model-145/predictions.npz')),'source_sha256':i.sha(Path(__file__))};i.write(_repo_path(ROOT, 'run/test-comparison.json'),r)
 meta=i.read(_repo_path(ROOT, 'test-run/test-feature-validation.json'));plan=i.read(_repo_path(ROOT, 'test-run/test-feature-plan.json'));i.write(_repo_path(ROOT, 'config/original-test-hashes.json'),{'CSV_sha256':plan['raw_CSV_sha256'],'image_sha256':meta['image_sha256'],'source':'Fresh176 raw1000 extraction, after full500 validation. Reference hashes for future exact input reproduction; no test fitting.'});print(json.dumps(r),flush=True)
if __name__=='__main__':main()
