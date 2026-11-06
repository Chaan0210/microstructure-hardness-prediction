"""Verify frozen covariance features, saved folds and preregistered diagnostics."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import token_cov_experiment_056 as e
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
fv=e.fv;OUT=e.OUT

def main():
 t=time.perf_counter();plan=e.check();fixtures=e.tests.run();ex=json.loads((OUT/'extraction.json').read_text());assert fv.sha(OUT/'development_features.npz')==ex['feature_sha256'];assert fv.sha(OUT/'projection.npy')==plan['projection_sha256'];R=np.load(_repo_io(OUT/'projection.npy'));np.testing.assert_array_equal(R,e.f.projection())
 with np.load(_repo_io(OUT/'development_features.npz')) as z:ids=z['ids'].tolist();X=z['X'].copy();assert z['names'].tolist()==list(e.f.NAMES)
 assert ids==e.ids_only() and X.shape==(400,528) and np.isfinite(X).all();torch.set_num_threads(2);m=e.ds.base.model().to('mps');replay=[]
 for j in (0,199,399):
  a,_,_=e.evaluate(m,ids[j],'mps',R);np.testing.assert_array_equal(a,X[j]);replay.append(float(abs(a-X[j]).max()));assert fv.sha(_repo_path(ROOT, 'data/train', f'{ids[j]}.png'))==ex['image_sha256'][ids[j]]
 del m
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz')) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 results={}
 for split in ('original','20261031','20261101'):
  out=OUT/split
  if not out.exists():continue
  report=json.loads((out/'report.json').read_text());pfile=out/'development_predictions.npz';assert fv.sha(pfile)==report['predictions_sha256']
  with np.load(_repo_io(pfile)) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);fid=z['fold'].copy();pred={n:z[n].copy() for n in report['scores']}
  src=_repo_path(ROOT, 'artifacts/seed-stability-052', split, 'development_oof.csv');assert fv.sha(src)==report['reference_oof_sha256']
  with src.open() as file:rows=list(csv.DictReader(file))
  assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(fid,[int(r['fold']) for r in rows]);np.testing.assert_array_equal(pred['reference007'],[float(r['candidate007_design']) for r in rows]);errors=[];extra=[]
  for k,log in enumerate(report['folds']):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261225+k).split(X[tr],y[tr],groups[tr]))
   for (a,b),(it,iv) in zip(log['inner_splits'],inner):np.testing.assert_array_equal(a,tr[it]);np.testing.assert_array_equal(b,tr[iv]);assert not set(groups[a])&set(groups[b]);assert set(a)|set(b)==set(tr)
   assert log['chosen']==min(log['inner_scores'],key=lambda q:q['rmse']);mp=out/f'fold-{k}.joblib';assert fv.sha(mp)==log['model_sha256'];saved=joblib.load(_repo_io(mp));model=saved['model'];assert saved['train_ids']==[ids[j] for j in tr] and saved['validation_ids']==[ids[j] for j in va];assert model.alpha==log['chosen']['alpha'];assert model.scaler.n_samples_seen_==320;assert model.ymean==float(y[tr].mean()) and model.ystd==float(y[tr].std());np.testing.assert_allclose(model.scaler.mean_,X[tr].mean(0),atol=1e-14,rtol=0);np.testing.assert_allclose(model.scaler.var_,X[tr].var(0),atol=1e-14,rtol=0);p=model.predict(X[va]);np.testing.assert_array_equal(p,pred['cov528'][va]);errors.append(float(abs(p-pred['cov528'][va]).max()));extra.append({'fold':k,'alpha':model.alpha,'validation_prediction_range':[float(p.min()),float(p.max())],'outside_foldtrain_label_range':int(((p<y[tr].min())|(p>y[tr].max())).sum()),'feature_cells_outside_foldtrain_range_fraction':float(((X[va]<X[tr].min(0))|(X[va]>X[tr].max(0))).mean())})
  np.testing.assert_array_equal(pred['primary007'],.9*pred['reference007']+.1*pred['cov528'])
  for n,p in pred.items():assert fv.rmse(y,p)==report['scores'][n]['rmse'];assert [fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]==report['scores'][n]['fold_rmse']
  c=report['comparison'];assert c['gain']==fv.rmse(y,pred['reference007'])-fv.rmse(y,pred['primary007']);fg=[fv.rmse(y[fid==k],pred['reference007'][fid==k])-fv.rmse(y[fid==k],pred['primary007'][fid==k]) for k in range(5)];assert fg==c['fold_gains'];ix=np.random.default_rng(20261226).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-pred['reference007'])[ix]**2,1))-np.sqrt(np.mean((y-pred['primary007'])[ix]**2,1));assert float(boot.std())==c['bootstrap_sd'];np.testing.assert_array_equal(np.percentile(boot,[2.5,97.5]),c['conditional_ci95']);assert float(np.corrcoef(y-pred['reference007'],y-pred['cov528'])[0,1])==c['residual_correlation']
  results[split]={'scores':report['scores'],'comparison':c,'reload_max_error':max(errors),'models_verified':len(errors),'extrapolation':extra,'residual_correlation_with_DINO_MLPmean3':float(np.corrcoef(y-pred['cov528'],y-np.array([float(r['mean3']) for r in rows]))[0,1])}
  if split=='original':original_fid=fid
 sel=json.loads((OUT/'selection.json').read_text());c=results['original']['comparison'];assert sel['advance']==bool(c['gain']>c['bootstrap_sd'] and c['positive_folds']>=3)
 if not sel['advance']:assert set(results)=={'original'}
 source=_repo_path(ROOT, 'artifacts/dino-multilayer-044/development_features.npz');assert fv.sha(source)=='6d26b7d78eaf1d094e829984d5665a38dc11b71c918f108384861259b35c4159'
 with np.load(_repo_io(source)) as z:assert z['ids'].tolist()==ids;A=z['X'][:,2].astype(float)
 error=np.zeros(528);null=np.zeros(528)
 for k in range(5):
  tr=np.flatnonzero(original_fid!=k);va=np.flatnonzero(original_fid==k);sx=StandardScaler().fit(A[tr]);sy=StandardScaler().fit(X[tr]);reg=Ridge(alpha=1152.).fit(sx.transform(A[tr]),sy.transform(X[tr]));truth=sy.transform(X[va]);estimate=reg.predict(sx.transform(A[va]));error+=((truth-estimate)**2).sum(0);null+=(truth**2).sum(0)
 valid=null>0;r2=1-error[valid]/null[valid];ratios=[a['cross_channel_contribution_relative_frobenius'] for a in ex['metadata']];diag={'meanstd_CLS_reconstruction_oof_R2_quantiles':np.quantile(r2,[0,.25,.5,.75,1]).tolist(),'meanstd_CLS_reconstruction_R2_median':float(np.median(r2)),'crosschannel_relative_Frobenius_quantiles':np.quantile(ratios,[0,.25,.5,.75,1]).tolist(),'minimum_eigenvalue_all_images':min(a['min_eigenvalue'] for a in ex['metadata']),'maximum_asymmetry':max(a['symmetry_max_error'] for a in ex['metadata']),'constant_columns':int((np.ptp(X,axis=0)==0).sum()),'interpretation':'Crosschannel magnitude is not independent information or target utility. Reconstruction is descriptive 5fold Ridge1152, all scalers trainfold-only; no feature selection. CLS plusmean/std control includes context.','position_limit':'Permutation invariant over output tokens. Input arrangement still influences backbone positional/contextual tokens.'}
 e.check();result={'passed':True,'fixtures':fixtures,'raw_feature_replay_max_error':max(replay),'raw_replay_rows':3,'projection_sha256':plan['projection_sha256'],'protected_files_verified':len(plan['protected']),'results':results,'diagnostics':diag,'selection':sel,'seconds':time.perf_counter()-t,'independent_code_review':'audit_pipeline no blocking math/leakage issue; planned reconstruction completed here','scope':'Development400 only. No holdout/test, candidateCSV, upload, newdata/packages/cost. Reused development selection bias persists.'};fv.write(_repo_path(ROOT, 'reports/token-cov-056-validation.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ('fixtures','results')}),flush=True)
if __name__=='__main__':main()
