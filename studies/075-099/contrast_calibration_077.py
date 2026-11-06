"""One additional signed component contrast in fully nested affine calibration."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import residual_diagnostics_076 as diag
import numpy as np
base=diag.base;fv=base.fv;OUT=_repo_path(ROOT, 'artifacts/contrast-calibration-077')

def check():
 base.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for path,digest in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, path))==digest,path
 return p

def prepare():
 base.check();pub=json.loads((_repo_path(ROOT, 'reports/candidate009-public-submission.json')).read_text());r=json.loads((diag.OUT/'report.json').read_text());assert sum(f['training_only_partial_correlation']>0 for s in r['results'].values() for f in s['folds'])==14;v=json.loads((_repo_path(ROOT, 'reports/nested-calibration-074-validation.json')).read_text());assert v['passed'] and v['eligible_for_candidate'];OUT.mkdir(exist_ok=False)
 p={'hypothesis':'A globally calibrated008 score still leaves residuals associated with whether DINOmean predicts above legacy003. One signed model-contrast term may correct relative reliability while preserving equal S/B contribution.',
 'single_primary':'OLS y=a+b*p008+c*(0.5*(Smean3+Bmean3)-legacy003). Three parameters total versus009 two. Center/scale both predictors using only320 inner-OOF training pairs, then unregularized least squares with intercept. No clipping, nonlinear knot, ratio grid, sign restriction, ridge search, component-specific S/B weights or Public tuning.',
 'training_data':'Reuse the45 verified074 inner cells. They refit every upstream estimator/scaler/kernel selection only inside its current outertrain. Combine each heldout-inner row predictions into p008 and signed contrast. Never fit calibrator on global outerOOF values. Outerfeatures are used only at the heldout80 rows.',
 'control':'Recompute original one-variable affine calibrator from identical inner320pairs; it must reproduce009 to1e-10. Any failure invalidates experiment.',
 'diagnostic_basis':'076: outer009 residual/contrast Pearson0.189/0.137/0.108 and training-only partialcorrelation positive14/15 folds. Diagnostics are adaptive and same400rows reused. These are not15independentconfirmations.',
 'screen':'OriginalpooledOOF strictly improves009 -> bothknownresplits unchanged; otherwise reject. Foldcounts/bootstrap diagnostic only.',
 'confirmation':'All3pooledOOF strictly improve009, exact control and strict upstream+metamodel leakage/reload verification -> lock final recipe and fit new coefficients on all500 nested OOF before testinference.',
 'nonduplication':'074 only calibrated aggregate scale/offset. 047 joint image-feature kernels and064 fixedS/Bmean did not learn a train-only legacy/DINO contrast coefficient. This is not another globalaffine rescaling or Public ratio optimization.',
 'risk':'Component predictors may be collinear and reliability may depend ontraining size. Reportconditionnumbers and effectivecoefficients, but do not choosemodels/constraints from those diagnostics. The inner213/214-to-outer320 training-size mismatch remains.',
 'resource':'ExistingCPU2; no upstream retraining for fitting because inputs andlearner unchanged. One wholeupstream cell replay withouterlabelperturbation verifiesnewcontrast aswellasaggregate. No newpackages/weights/cost.',
 'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['contrast_calibration_077.py','validate_contrast_calibration_077.py','residual_diagnostics_076.py','nested_calibration_074.py','validate_nested_calibration_074_v2.py']},
 'inputs':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/residual-diagnostics-076/report.json','artifacts/residual-diagnostics-076/inner_training_matrices.npz','reports/nested-calibration-074-validation.json','reports/candidate009-public-submission.json']},
 'protected':{**base.check()['protected'],**pub['protected_sha256']},'scope':'Development400onlyduringselection. Final500 artifacts onlyhashed untilvalidatedrecipeislocked. Noholdout/testmodelselection.','bias':fv.CAVEAT}
 fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'hypothesis':p['hypothesis'],'single_primary':p['single_primary']}),flush=True)

def fit_meta(A,y):
 A=np.asarray(A,dtype=float);y=np.asarray(y,dtype=float);assert A.shape==(len(y),2) and np.isfinite(A).all() and np.isfinite(y).all();mean=A.mean(0);scale=A.std(0);assert (scale>0).all();Z=(A-mean)/scale;target=float(y.mean());coef,_,rank,singular=np.linalg.lstsq(Z,y-target,rcond=None);assert rank==2 and np.isfinite(coef).all();slopes=coef/scale;intercept=target-float(mean@slopes)
 return {'mean':mean.tolist(),'scale':scale.tolist(),'target_mean':target,'standardized_coefficients':coef.tolist(),'raw_slopes':slopes.tolist(),'intercept':intercept,'rank':int(rank),'condition_number':float(singular.max()/singular.min()),'n_train':len(y),'equivalent_raw_prediction_coefficients':{'legacy003':float(.75*slopes[0]-slopes[1]),'Smean3':float(.125*slopes[0]+.5*slopes[1]),'Bmean3':float(.125*slopes[0]+.5*slopes[1])}}

def predict_meta(m,A):return m['target_mean']+(np.asarray(A)-np.array(m['mean']))/np.array(m['scale'])@np.array(m['standardized_coefficients'])

def run(split):
 check()
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 d=diag.load_outer(split);y=d['y'];fid=d['fid'];ref=d['reference009'];out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full(400,np.nan);control=np.full(400,np.nan);folds=[];start=time.perf_counter();outerA=np.column_stack([d['reference008'],d['contrast']])
 for k in range(5):
  tr,va,A,C=diag.inner_matrix(d,split,k);fd=out/f'fold-{k}';fd.mkdir();m=fit_meta(A,y[tr]);m.update(train_ids=[d['ids'][i] for i in tr],validation_ids=[d['ids'][i] for i in va],source_inner_fold_directory=str((base.OUT/split/f'fold-{k}').relative_to(ROOT)));fv.write(fd/'model.json',m);pred[va]=predict_meta(m,outerA[va]);pc=A[:,0]-A[:,0].mean();b=float(pc@(y[tr]-y[tr].mean())/(pc@pc));control[va]=y[tr].mean()+b*(d['reference008'][va]-A[:,0].mean());np.testing.assert_allclose(control[va],ref[va],atol=1e-10,rtol=0)
  with (fd/'fit_data.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,inner_A=A,outer_A=outerA[va],predictions=pred[va],control009=control[va])
  rec={'fold':k,'model_sha256':fv.sha(fd/'model.json'),'fit_data_sha256':fv.sha(fd/'fit_data.npz'),'raw_slopes':m['raw_slopes'],'condition_number':m['condition_number'],'effective_coefficients':m['equivalent_raw_prediction_coefficients'],'control009_max_error':float(abs(control[va]-ref[va]).max()),'outer_rmse':fv.rmse(y[va],pred[va])};fv.write(fd/'report.json',rec);folds.append(rec)
 comp=base.prior.old.comparison(y,ref,pred,fid)
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=y,fold=fid,reference009=ref,primary=pred,control009=control)
 report={'split':split,'scores':{n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in [('reference009',ref),('primary',pred)]},'comparison':comp,'folds':folds,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds':time.perf_counter()-start,'upstream_models_reused_not_refitted':True,'candidate_created':False};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(comp['gain']>0),'comparison':comp})
 check();print(json.dumps({k:v for k,v in report.items() if k!='folds'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','original','20261031','20261101']);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
