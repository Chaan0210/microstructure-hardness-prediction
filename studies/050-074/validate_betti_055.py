"""Saved-model audit and fixed descriptive diagnostics; no feature/model selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import betti_experiment_055 as e
import numpy as np,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
fv=e.fv;OUT=e.OUT

def main():
 t=time.perf_counter();plan=e.check();fixtures=e.tests.run();assert fixtures['passed']
 ex=json.loads((OUT/'extraction.json').read_text());assert fv.sha(OUT/'development_features.npz')==ex['feature_sha256']
 with np.load(_repo_io(OUT/'development_features.npz')) as z:ids=z['ids'].tolist();X=z['X'].copy();names=z['names'].tolist()
 assert ids==e.ids_only() and X.shape==(400,132) and len(set(names))==132 and np.isfinite(X).all()
 for j in (0,199,399):np.testing.assert_array_equal(e.f.extract(e.image(ids[j]))[0],X[j]);assert fv.sha(_repo_path(ROOT, 'data/train', f'{ids[j]}.png'))==ex['image_sha256'][ids[j]]
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz')) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 results={}
 for split in ('original','20261031','20261101'):
  out=OUT/split
  if not out.exists():continue
  report=json.loads((out/'report.json').read_text());path=out/'development_predictions.npz';assert fv.sha(path)==report['predictions_sha256']
  with np.load(_repo_io(path)) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);fid=z['fold'].copy();pred={n:z[n].copy() for n in report['scores']}
  src=_repo_path(ROOT, 'artifacts/seed-stability-052', split, 'development_oof.csv');assert fv.sha(src)==report['reference_oof_sha256']
  with src.open() as file:rows=list(csv.DictReader(file))
  assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(fid,[int(r['fold']) for r in rows]);np.testing.assert_array_equal(pred['reference007'],[float(r['candidate007_design']) for r in rows])
  errors=[]
  for log in report['folds']:
   arm=log['arm'];k=log['fold'];tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va]);cols=np.arange(132) if arm=='betti132' else e.f.ZERO_COLUMNS;A=X[:,cols]
   inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261223+k).split(A[tr],y[tr],groups[tr]))
   for (a,b),(it,iv) in zip(log['inner_splits'],inner):np.testing.assert_array_equal(a,tr[it]);np.testing.assert_array_equal(b,tr[iv]);assert not set(groups[a])&set(groups[b]);assert set(a)|set(b)==set(tr)
   assert log['chosen']==min(log['inner_scores'],key=lambda q:q['rmse'])
   path=out/f'{arm}-fold-{k}.joblib';assert fv.sha(path)==log['model_sha256'];saved=joblib.load(_repo_io(path));m=saved['model'];assert saved['train_ids']==[ids[j] for j in tr] and saved['validation_ids']==[ids[j] for j in va];np.testing.assert_array_equal(saved['columns'],cols);assert m.alpha==log['chosen']['alpha'];assert m.scaler.n_samples_seen_==320
   np.testing.assert_allclose(m.scaler.mean_,A[tr].mean(0),rtol=0,atol=1e-14);np.testing.assert_allclose(m.scaler.var_,A[tr].var(0),rtol=0,atol=1e-14);assert m.ymean==float(y[tr].mean()) and m.ystd==float(y[tr].std());replay=m.predict(A[va]);np.testing.assert_array_equal(replay,pred[arm][va]);errors.append(float(abs(replay-pred[arm][va]).max()))
  for arm in ('betti132','counts12'):np.testing.assert_array_equal(pred[arm+'_blend'],.9*pred['reference007']+.1*pred[arm])
  for n,p in pred.items():assert fv.rmse(y,p)==report['scores'][n]['rmse'];assert [fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]==report['scores'][n]['fold_rmse']
  c=report['comparison'];gain=fv.rmse(y,pred['reference007'])-fv.rmse(y,pred['betti132_blend']);assert gain==c['gain'];fg=[fv.rmse(y[fid==k],pred['reference007'][fid==k])-fv.rmse(y[fid==k],pred['betti132_blend'][fid==k]) for k in range(5)];assert fg==c['fold_gains'];ix=np.random.default_rng(20261224).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-pred['reference007'])[ix]**2,1))-np.sqrt(np.mean((y-pred['betti132_blend'])[ix]**2,1));assert float(boot.std())==c['bootstrap_sd'];np.testing.assert_array_equal(np.percentile(boot,[2.5,97.5]),c['conditional_ci95'])
  results[split]={'scores':report['scores'],'comparison':c,'reload_max_error':max(errors),'models_verified':len(errors),'predictions_sha256':report['predictions_sha256'],'extrapolation':[{'arm':r['arm'],'fold':r['fold'],'label_range_exceedances':r['validation_outside_train_label_range'],'feature_cells_outside_train_range_fraction':r['validation_feature_cells_outside_train_range_fraction'],'prediction_range':r['prediction_range'],'chosen_alpha':r['chosen']['alpha']} for r in report['folds']]}
  if split=='original':original_fid=fid.copy()
 selection=json.loads((OUT/'selection.json').read_text());c=results['original']['comparison'];assert selection['advance']==bool(c['gain']>c['bootstrap_sd'] and c['positive_folds']>=3)
 if not selection['advance']:assert set(results)=={'original'}
 # Diagnostic reconstruction: can zero-distance counts predict the full curves?
 # Both input and output scaling fit each outertrain; no hardness labels used.
 remaining=np.array([j for j in range(132) if j not in e.f.ZERO_COLUMNS]);residual_ss=np.zeros(len(remaining));null_ss=np.zeros(len(remaining));scale_exceptions=0
 for k in range(5):
  tr=np.flatnonzero(original_fid!=k);va=np.flatnonzero(original_fid==k);sx=StandardScaler().fit(X[tr][:,e.f.ZERO_COLUMNS]);sy=StandardScaler().fit(X[tr][:,remaining]);model=Ridge(alpha=120.).fit(sx.transform(X[tr][:,e.f.ZERO_COLUMNS]),sy.transform(X[tr][:,remaining]));truth=sy.transform(X[va][:,remaining]);estimate=model.predict(sx.transform(X[va][:,e.f.ZERO_COLUMNS]));residual_ss+=((truth-estimate)**2).sum(0);null_ss+=(truth**2).sum(0);scale_exceptions+=int(((sy.var_==0)&(np.ptp(X[va][:,remaining],axis=0)>0)).sum())
 valid=null_ss>0;r2=1-residual_ss[valid]/null_ss[valid];constant=np.ptp(X,axis=0)==0;unique=len(np.unique(X.T,axis=0));hf=np.array([m['high_frequency_rms'] for m in ex['metadata']]);cc=np.corrcoef(X[:,~constant].T,hf)[-1,:-1];nonconstant_names=np.array(names)[~constant];rank=np.argsort(abs(cc))[-5:][::-1]
 diag={'constant_columns':int(constant.sum()),'unique_column_vectors':unique,'duplicate_columns':132-unique,'d0_d1_exactly_duplicate':bool(np.array_equal(X.reshape(400,3,11,2,2)[:,:,5],X.reshape(400,3,11,2,2)[:,:,6])),'zero_count_reconstruction_r2_median':float(np.median(r2)),'zero_count_reconstruction_r2_quartiles':np.quantile(r2,[0,.25,.5,.75,1]).tolist(),'reconstruction_fold_constant_train_varying_validation_columns':scale_exceptions,'reconstruction_limit':'Descriptive OOF standardized-error R2 vs outertrain-mean null, not independent validation or feature selection. Some columns nearly constant.','highfrequency_abs_correlation_median':float(np.median(abs(cc))),'highfrequency_top_correlations':[{'feature':str(nonconstant_names[j]),'correlation':float(cc[j])} for j in rank],'highfrequency_limit':'Image high-frequency RMS mixes noise, sharp boundaries and contrast; correlation cannot distinguish cause. Not fed to regressor.','granulometry_distinction':'Existing grayscale opening/closing intensity statistics differ from components and holes through signed-distance excursions; thin/thick neck fixture has identical zero-level Betti counts but different eroded connectivity. Does not prove empirical information independent of every prior descriptor.'}
 e.check();result={'passed':True,'fixtures':fixtures,'development_rows':400,'features':132,'raw_replay_rows':3,'protected_artifacts_verified':len(plan['protected']),'results':results,'diagnostics':diag,'selection':selection,'validation_seconds':time.perf_counter()-t,'scope':'No holdout/test images or labels; no candidateCSV,upload,install,cost. Model selection only preregistered original screen; repeated-development bias remains.'};fv.write(_repo_path(ROOT, 'reports/betti-055-validation.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ('fixtures','results')}),flush=True)
if __name__=='__main__':main()
