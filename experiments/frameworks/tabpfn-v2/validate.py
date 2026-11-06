from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import run as r
import numpy as np,joblib,gc
R=r.R;P=r.P

def main():
 p=r.check();counts=0;pca_errors=[];all_reports={}
 with r.Monitor('verification','verification'):
  with np.load(_repo_io(P/'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
  with np.load(_repo_io(P/'experiments/dinol-177/features.npz'),allow_pickle=False) as z:D=z['X'].astype(float);np.testing.assert_array_equal(c['ids'],z['ids'])
  with np.load(_repo_io(P/'experiments/dinol-177/original/predictions.npz'),allow_pickle=False) as z:g={k:z[k].copy() for k in z.files}
  np.testing.assert_array_equal(c['y'],g['y']);np.testing.assert_array_equal(c['groups'],g['g'])
  for rep in p['representations']:
   root=R/'results'/rep;report=r.read(root/'report.json');assert r.sha(root/'OOF.npz')==report['OOF_sha256'];rebuilt=np.full(400,np.nan)
   for k in range(5):
    out=root/f'fold-{k}';fr=r.read(out/'report.json');assert r.sha(out/'report.json')==report['fold_report_sha256'][str(k)];assert r.sha(out/'estimator.joblib')==fr['model_sha256'];assert r.sha(out/'predictions.npz')==fr['predictions_sha256']
    with np.load(_repo_io(out/'predictions.npz'),allow_pickle=False) as z:T=z['train_indices'];V=z['validation_indices'];pred=z['prediction'];np.testing.assert_array_equal(z['ids'],c['ids'][V])
    np.testing.assert_array_equal(T,np.flatnonzero(g['fold']!=k));np.testing.assert_array_equal(V,np.flatnonzero(g['fold']==k));assert len(T)==320 and len(V)==80 and not set(c['groups'][T])&set(c['groups'][V]);rebuilt[V]=pred
    est=joblib.load(_repo_io(out/'estimator.joblib'));params=est.get_params(deep=False);assert params['device']=='cpu' and params['fit_mode']=='low_memory' and params['n_estimators']==params['n_preprocessing_jobs']==1;assert str(params['inference_precision'])=='torch.float32';assert params['random_state']==202710184+k;assert type(params['model_path']).__name__=='ModelSpecs';assert type(params['model_path'].architecture_config).__module__.endswith('tabpfn_v2.config') or 'tabpfn_v2' in type(params['model_path'].architecture_config).__module__;assert est.executor_.X_train.shape[0]==320
    if rep=='compact26':expected=c['X'][T].astype(np.float32)
    else:
     obj=joblib.load(_repo_io(R/'PCA'/f'fold-{k}.joblib'));sc=obj['scaler'];pc=obj['PCA'];np.testing.assert_array_equal(obj['train_indices'],T);np.testing.assert_array_equal(obj['validation_indices'],V);np.testing.assert_allclose(sc.mean_,D[T].mean(0),atol=1e-12,rtol=0);np.testing.assert_allclose(sc.var_,D[T].var(0),atol=1e-12,rtol=0);assert pc.n_samples_==320
     Z=(D[T]-sc.mean_)/sc.scale_;S=(Z-pc.mean_)@pc.components_.T;err=float(abs(S.var(0,ddof=1)-pc.explained_variance_).max());pca_errors.append(err);assert err<1e-8;np.testing.assert_allclose(pc.components_@pc.components_.T,np.eye(64),atol=1e-12)
     expected=(S if rep=='DINO_L_PCA64' else np.column_stack([c['X'][T],S])).astype(np.float32)
    actual=np.asarray(est.executor_.X_train);np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-6);assert all(v.device.type=='cpu' for v in est.models_[0].parameters());counts+=1;del est;gc.collect()
   with np.load(_repo_io(root/'OOF.npz'),allow_pickle=False) as z:
    np.testing.assert_array_equal(z['ids'],c['ids']);np.testing.assert_array_equal(z['y'],c['y']);np.testing.assert_array_equal(z['fold'],g['fold']);np.testing.assert_array_equal(z['best021'],g['blend_L_MLP']);np.testing.assert_array_equal(z['standalone'],rebuilt);np.testing.assert_array_equal(z['blend05'],.95*g['blend_L_MLP']+.05*rebuilt)
    for n in ['best021','standalone','blend05']:assert abs(float(np.sqrt(np.mean((c['y']-z[n])**2)))-report['RMSE'][n])<1e-12
   all_reports[rep]=r.sha(root/'report.json')
  current={str(q.relative_to(P)):r.sha(q) for q in (P/'.venv/lib/python3.13/site-packages').glob('*.dist-info/METADATA')};assert current==p['original_environment_metadata_sha256'];r.check();assert len(r.network_events)==0
  rec={'passed':True,'saved_estimators_independently_inspected':counts,'all15_CPU_1estimator_float32_lowmemory_ModelSpecs_v2':True,'all15_training_feature_matrices_verified':True,'PCA64_train320_variance_and_orthogonality_max_error':max(pca_errors),'all3_OOF_ID_target_fold_group_alignment_and_metrics_exact':True,'existing_environment_all_distribution_metadata_unchanged':True,'protected_files_unchanged':len(p['protected']),'report_sha256':all_reports,'no_extra_fit_or_predict_in_this_validation':True,'network_attempts':len(r.network_events)};r.write(R/'validation.json',rec);r.emit(rec)
if __name__=='__main__':main()
