"""Independent artifact checks and diagnostic report, development cache only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import numpy as np,joblib
import learning_curve_050 as c

def main():
 start=time.perf_counter();p=json.loads((c.OUT/'plan.json').read_text());r=json.loads((c.OUT/'report.json').read_text())
 with np.load(_repo_path(ROOT, 'artifacts/dino-multilayer-044/development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();X=z['X'].astype(float).reshape(400,3456)
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'];groups=z['groups']
 manifest=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());fid=np.array([manifest['development_folds'][i] for i in ids])
 with (c.OUT/'validation_predictions.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==400 and [a['ID'] for a in rows]==ids;assert np.array_equal([int(a['fold']) for a in rows],fid)
 errors=[];train={80:[],160:[],320:[]}
 for f in range(5):
  val=np.flatnonzero(fid==f);order=np.random.default_rng(c.SEED).permutation(np.flatnonzero(fid!=f));last=set()
  for n in (80,160,320):
   path=c.OUT/f'fold-{f}-n{n}.joblib';a=joblib.load(_repo_io(path));sub=order[:n];assert a['train_ids']==[ids[j] for j in sub];assert a['validation_ids']==[ids[j] for j in val];assert last<=set(sub);last=set(sub);assert not set(groups[sub])&set(groups[val]);m=a['model']
   assert m.alpha==3456.;assert m.kind=='all_ridge';assert m.pca is None;assert m.scaler.n_samples_seen_==n
   np.testing.assert_allclose(m.scaler.mean_,X[sub].mean(0),rtol=0,atol=1e-12);np.testing.assert_allclose(m.scaler.var_,X[sub].var(0),rtol=1e-12,atol=1e-12);assert m.ymean==float(y[sub].mean());assert m.ystd==float(y[sub].std())
   vp=m.predict(X[val]);saved=np.array([float(rows[j][f'n{n}']) for j in val]);error=float(abs(vp-saved).max());assert error==0;errors.append(error);train[n].extend((m.predict(X[sub])-y[sub]).tolist())
   rec=next(v for v in r['records'] if v['fold']==f and v['n_train']==n);assert c.sha(path)==rec['model_sha256']
 for n in train:
  point=next(v for v in r['points'] if v['n_train']==n);v=np.array([float(a[f'n{n}']) for a in rows]);assert np.isfinite(v).all();assert abs(c.rmse(y,v)-point['pooled_validation_rmse'])<1e-12;assert abs(float(np.sqrt(np.mean(np.array(train[n])**2)))-point['pooled_train_rmse'])<1e-12
 for path,h in {**p['input_sha256'],**p['source_sha256'],**p['protected']}.items():assert c.sha(_repo_path(ROOT, path))==h,path
 out={'valid':True,'n_models':15,'n_validation_predictions_each_size':400,'max_reload_prediction_error':max(errors),'nested_group_disjoint_subsets_verified':True,'subset_only_X_y_transforms_verified':True,'fixed_alpha_verified':True,'pooled_train_validation_rmse_recalculated':True,'cache_and_best006_hashes_unchanged':True,'report_sha256':c.sha(c.OUT/'report.json'),'plan_sha256':c.sha(c.OUT/'plan.json'),'seconds':time.perf_counter()-start}
 c.save(_repo_path(ROOT, 'reports/learning-curve-050-validation.json'),out)
 text='Learning curve050: completed diagnostic, no model selection.\n\n'
 text+='Scope: development400; cached frozen006 DINO3456 branch only, not full006 ensemble. Same outer5folds; one subset ordering seed; nested80/160/320 train with80 fixed validation per fold.\n'
 text+='Setting: all_ridge alpha3456; StandardScaler and target mean/std fit current subset only. No PCA, inner search, new image extraction, holdout/test access, package install or submission.\n\n'
 text+='Train size | training RMSE | validation RMSE | gap\n'
 for x in r['points']:text+=f"{x['n_train']:3d} | {x['pooled_train_rmse']:.4f} | {x['pooled_validation_rmse']:.4f} | {x['generalization_gap']:.4f}\n"
 text+='\n160 to320: pooled validation gain0.8276; five fold gains0.8341,0.9974,1.1208,0.5190,0.6638 (5/5 positive). Fold gain SD0.2434 is descriptive; overlapping training sets are not independent replications.\n'
 text+='Interpretation: consistent benefit from more samples in this fixed condition; train/validation gap narrows but remains. No observed saturation or identified noise floor. One subset seed and reused400 rows limit generality. Fixed alpha also changes effective alpha/n with size, so sample count is not the only changed effective quantity.\n'
 text+='Training RMSE near11 is not an irreducible lower bound: prior044 weaker regularization (lambda.1) on folds0/2 reached training RMSE5.657/5.580 with mixed validation effect. Capacity, shrinkage bias, representation quality and label noise remain unresolved.\n'
 text+='The submitted006 already used all500 labels; simply repeating full500 fitting is not a new improvement. Do not extrapolate this320-point curve to an expected leaderboard gain.\n\n'
 text+='Next options, not executed:\n'
 text+='1. Existing approved environment: one matched diagnostic with alpha scaled by n/320, preserving alpha/n, would isolate part of the regularization confound. Same15 small CPU fits; measured present15fit/check loop0.179s excluding imports/cache load. This is a diagnostic, not a sweep or new submission.\n'
 text+='2. Existing environment model work: this curve alone does not identify a new feature, nonlinear model or layer weighting that should improve. Previously failed kernel/PLS/morphology families should not be repeated without a distinct cause. More precise official generator/label information could inform a better hypothesis; not fetched or requested from others here.\n'
 text+='3. New installation: no new package is justified by this curve. Additional learners would require prior installation approval; the current evidence does not show why they should improve over existing sklearn/torch. External training data remain outside scope.\n\n'
 text+=f"Measured fit/predict/serialization loop {r['seconds']:.3f}s (excludes imports/cache reads); peak RSS {r['peak_rss_bytes']/1e6:.1f}MB.15 model reloads exact; matched prior lambda1 full-size predictions differ by <= {r['matching_existing_full_subset_prediction_max_error']:.3g}.\n"
 text+='Best006 CSV/design/model unchanged. No candidate, submission, cost or running work.\n'
 text+='Commands: .venv/bin/python -I -B learning_curve_050.py ; .venv/bin/python -I -B validate_learning_curve_050.py (creation is exclusive; rerun in a fresh project-contained output copy).\n'
 (_repo_path(ROOT, 'reports/learning-curve-050-results.txt')).write_text(text)
 sfile=_repo_path(ROOT, 'reports/status.json');s=json.loads(sfile.read_text());from datetime import datetime,timezone
 s.update(status='learning_curve050_completed_diagnostic_only',latest_completed='reports/learning-curve-050-summary.json',latest_diagnostic='reports/learning-curve-050-summary.json',latest_diagnostic_oof_rmse=None,latest_diagnostic_is_independent_validation=False,latest_verification='reports/learning-curve-050-validation.json',running_work=[],new_submission_candidate=None,current_study_submission_performed=False,current_study_cost=0,updated_at_utc=datetime.now(timezone.utc).isoformat());sfile.write_text(json.dumps(s,indent=2))
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
