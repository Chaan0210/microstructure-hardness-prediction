"""One fixed train-only scaling repair of preserved persistence study107."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np,joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import persistence_study_107 as old
import amplitude_cnn_106 as a
OUT=_repo_path(ROOT, 'artifacts/persistence-repair-109');ARMS=old.ARMS;SPLITS=old.SPLITS;ALPHAS=old.ALPHAS;read=old.read;write=old.write;sha=old.sha;emit=old.emit;rmse=old.rmse
class TrainFloorScaler:
 def fit(self,X):
  X=np.asarray(X,dtype=float);self.mean_=X.mean(0);self.var_=X.var(0);self.std_=np.sqrt(self.var_);self.floor_=float(np.sqrt(self.var_.mean()));self.scale_=np.maximum(self.std_,self.floor_) if self.floor_>0 else np.ones(X.shape[1]);self.n_samples_seen_=len(X);return self
 def transform(self,X):return (np.asarray(X,dtype=float)-self.mean_)/self.scale_
class Regressor:
 def __init__(self,alpha):self.alpha=alpha
 def fit(self,X,y):
  self.scaler=TrainFloorScaler().fit(X);self.ym=float(y.mean());self.ys=float(y.std());assert self.ys>0;self.ridge=Ridge(alpha=self.alpha,solver='cholesky').fit(self.scaler.transform(X),(y-self.ym)/self.ys);return self
 def predict(self,X):return self.ridge.predict(self.scaler.transform(X))*self.ys+self.ym
TrainFloorScaler.__module__=Regressor.__module__='persistence_repair_109'
if __name__=='__main__':sys.modules['persistence_repair_109']=sys.modules[__name__]

def check():
 p=read(_repo_path(OUT, 'plan.json'));a.timecheck();old.check()
 for n,h in {**p['sources'],**p['inputs'],**p['original107_preserved'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def prepare():
 OUT.mkdir(exist_ok=False);old.check();assert read(_repo_path(ROOT, 'reports/persistence-107-original-validation.json'))['passed'];src=['persistence_repair_109.py','validate_persistence_repair_109.py'];inp=['artifacts/persistence-107/development_features.npz','artifacts/persistence-107/extraction.json','artifacts/persistence-107/original/report.json','artifacts/persistence-107/failure-diagnosis.json','reports/persistence-107-original-validation.json','reports/research-batch-105-108-summary.json'];preserved=[*sorted(old.OUT.rglob('*')),*[_repo_path(ROOT, n) for n in ['persistence_features_107.py','persistence_study_107.py','validate_persistence_107.py','diagnose_persistence_107.py']]]
 p={'study':'109 one fixed statistical scaling repair of107','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':a.DEADLINE,'authorization':'Parent followup explicitlyauthorizes separating extremevariance scaling failure fromhypothesis failure; preserve107 andapplyone fixed train-only correction withinexisting2hbatch. No validation-based thresholdsearch.',
 'repair':'For each inner/outer trainingmatrix, mu_j=mean(X_j),var_j=populationvariance(X_j),s_RMS=sqrt(mean_j var_j). scale_j=max(sqrt(var_j),s_RMS). Transform(X_j-mu_j)/scale_j. Ifallvariancezero useunit scales. Keep512columns; no featuredeletion,clipping,PCA,cellmask,bin/sigma changes. Bothcurve andpersistence receiveidenticalrule.',
 'rationale':'Columns within each descriptor share physical units. An RMS feature-variance floor limits inversevariance amplification while retainingrelativeamplitudes ofweakercolumns. This is fixed feature-scaling regularization, not a provenoptimal varianceestimator. Every trainingcolumn variance afterscaling<=1; exactconstanttraincolumns havezero fittedRidge coefficient. No numericaltinyvalue ispromotedto unitvariance solelybecause itisnonzero. Rulehasno searchedcoefficient or validation-dependent cutoff.',
 'unchanged':'Reuse exact107 cachedraw features,IDs,target,outer/innergroupfolds,inneralpha51.2/512/5120,seed20271007+k,targetstandardization,Ridgecholeskyfit, and .9same011+.1branch. No extraction/relabeling/newimages. Original107models/resultsimmutable.',
 'preflight_before_OOF':'For eachoriginalouterfold andbotharms, compareoriginalsavedmodel againstrepairfit using SAME originalselectedalpha. Report train/validationfeaturedistribution,scalequantiles,effectiverank,regularizeddualgram condition number,predictionranges andcountsoutsidetrainlabelrange. Do not calculatevalidationRMSE,inspectvalidationlabels or chooseanything frompreflight. Fixedoldalphamodelsdiagnosticonly; finalnestedOOF independentlyselectsalpha byinnerCV.',
 'condition_definition':'A=centeredscaledtrainingmatrix; singularvalues s. Numericaldesignrank threshold=max(n,p)*eps*s_max. Rectangularcentereddesign isrankdeficient; do notcallfullconditionfinite. Report effectivepositive-spectrumcondition andcond(A A.T+alpha I)=(smax^2+alpha)/(smin^2+alpha). Distinguishtraininglinear-systemcondition fromvalidationextrapolation.',
 'OOF_rule':'Runfixedoriginal5fold afterpreflight. Primaryrepairedpersistenceblend muststrictlyimprovebest011 and repairedcurveblend foradvancement. Ifpassplusindependentvalidation, BOTH existingresplits unchanged andsamecriterionall3 beforefull500/test. Ifnotbetter011,close thispersistenceapproach perparent. No furtherfloor,mask,regularizer/weightsearch.',
 'verification':'IndependentRMSfloor/scaling anddualRidge calculation; allinnerCVscores/alphareselection replay; modelreload; outervalidationlabelperturbation fullnestedfold0botharms; feature/protected/original107hashes. No physicalstructurepreservation claim.',
 'scope':'CPU2existingvenv,cost0,noinstall,noextra rawimageextraction/holdout/test selection; oldartifactshashedforintegrityonly.','bias':'Thisfix followsobserved107failure andisadaptive. Original400andknownresplits are reused; correctiondoesnotmakeevidence independent.','sources':{n:sha(_repo_path(ROOT, n)) for n in src},'inputs':{n:sha(_repo_path(ROOT, n)) for n in inp},'original107_preserved':{str(p.relative_to(ROOT)):sha(p) for p in preserved if p.is_file()},'protected011':a.check()['protected011'],'cost':0};write(_repo_path(OUT, 'plan.json'),p);check();emit({'prepared':True,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'original107_files_preserved':len(p['original107_preserved'])})

def distribution(x):
 v=np.asarray(x,dtype=float);return {'min':float(v.min()),'max':float(v.max()),'abs_quantiles_50_90_99_100':np.quantile(abs(v),[.5,.9,.99,1]).tolist(),'finite':bool(np.isfinite(v).all())}
def diagnostics(model,X,y,Z):
 A=model.scaler.transform(X);B=model.scaler.transform(Z);C=A-A.mean(0);sv=np.linalg.svd(C,compute_uv=False);tol=max(C.shape)*np.finfo(float).eps*sv[0] if sv[0]>0 else 0.;rank=int((sv>tol).sum());nz=sv[sv>tol];pred=model.predict(Z);assert np.isfinite(pred).all()
 return {'raw_train_features':distribution(X),'raw_validation_features':distribution(Z),'train_std_quantiles_0_25_50_75_100':np.quantile(np.sqrt(model.scaler.var_),[0,.25,.5,.75,1]).tolist(),'applied_scale_quantiles_0_25_50_75_100':np.quantile(model.scaler.scale_,[0,.25,.5,.75,1]).tolist(),'RMS_floor':getattr(model.scaler,'floor_',None),'floored_columns':int(np.sum(np.sqrt(model.scaler.var_)<model.scaler.floor_)) if hasattr(model.scaler,'floor_') else 0,'scaled_train_features':distribution(A),'scaled_validation_features':distribution(B),'design_shape':list(C.shape),'numerical_rank':rank,'design_rank_deficient':rank<min(C.shape),'effective_positive_singular_condition':float(nz[0]/nz[-1]) if len(nz) else None,'regularized_dual_gram_condition':float((sv[0]**2+model.alpha)/(sv[-1]**2+model.alpha)),'alpha':model.alpha,'train_target_range':[float(y.min()),float(y.max())],'validation_predictions':distribution(pred),'predictions_outside_train_target_range':int(((pred<y.min())|(pred>y.max())).sum())},pred

def preflight():
 check();d=old.inputs('original');rows=[];saved={};start=time.perf_counter()
 for arm in ARMS:
  X=d[arm]
  for k in range(5):
   tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);orig=joblib.load(_repo_io(old.OUT/'original'/f'{arm}-fold{k}.joblib'))['model'];fixed=Regressor(orig.alpha).fit(X[tr],d['y'][tr]);r0,q0=diagnostics(orig,X[tr],d['y'][tr],X[va]);r1,q1=diagnostics(fixed,X[tr],d['y'][tr],X[va]);rows.append({'arm':arm,'fold':k,'same_fixed_original_alpha':orig.alpha,'original':r0,'repaired':r1});saved[f'{arm}_fold{k}_original']=q0;saved[f'{arm}_fold{k}_repaired']=q1;emit({'preflight_arm':arm,'fold':k,'original_max_abs_z':r0['scaled_validation_features']['abs_quantiles_50_90_99_100'][-1],'repaired_max_abs_z':r1['scaled_validation_features']['abs_quantiles_50_90_99_100'][-1],'original_prediction_range':[r0['validation_predictions']['min'],r0['validation_predictions']['max']],'repaired_prediction_range':[r1['validation_predictions']['min'],r1['validation_predictions']['max']]})
 with (_repo_path(OUT, 'preflight-predictions.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),**saved)
 report={'complete':True,'records':rows,'validation_RMSE_computed':False,'validation_y_passed_to_fit_or_diagnostics':False,'note':'Existing inputs loader verifiesy arrayalignment; diagnosticcode passesonlytrainlabels and neverreadsheldoutlabelvalues. Oldselectedalphaisheldfixedforthiscomparison; modelsnoteligiblealternatecandidates.','all_predictions_finite':True,'prediction_sha256':sha(_repo_path(OUT, 'preflight-predictions.npz')),'seconds':time.perf_counter()-start,'plan_sha256':sha(_repo_path(OUT, 'plan.json'))};write(_repo_path(OUT, 'preflight.json'),report);check();emit({'preflight_complete':True,'seconds':report['seconds'],'report_sha256':sha(_repo_path(OUT, 'preflight.json'))})

def nested(X,d,tr,va,k):
 scores=[];splits=list(GroupKFold(3,shuffle=True,random_state=20271007+k).split(X[tr],groups=d['g'][tr]))
 for alpha in ALPHAS:
  q=np.full(len(tr),np.nan)
  for it,iv in splits:
   assert not set(d['g'][tr[it]])&set(d['g'][tr[iv]]);m=Regressor(alpha).fit(X[tr[it]],d['y'][tr[it]]);q[iv]=m.predict(X[tr[iv]])
  scores.append({'alpha':alpha,'rmse':rmse(d['y'][tr],q)})
 chosen=min(scores,key=lambda z:z['rmse']);m=Regressor(chosen['alpha']).fit(X[tr],d['y'][tr]);return m.predict(X[va]),m,{'inner_scores':scores,'selected':chosen,'inner_indices':[[tr[a].tolist(),tr[b].tolist()] for a,b in splits]}

def run(split):
 check();assert read(_repo_path(OUT, 'preflight.json'))['complete']
 if split!='original':assert read(_repo_path(OUT, 'original/report.json'))['gate']['advance'] and read(_repo_path(ROOT, 'reports/persistence-repair-109-original-validation.json'))['passed']
 d=old.inputs(split);dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);start=time.perf_counter();ps={'reference011':d['ref']};mh={};selected={}
 for arm in ARMS:
  X=d[arm];pred=np.full(400,np.nan)
  for k in range(5):
   a.timecheck();tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);pred[va],m,detail=nested(X,d,tr,va,k);path=dest/f'{arm}-fold{k}.joblib';joblib.dump({'model':m,'detail':detail,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va]},_repo_io(path));mh[path.name]=sha(path);selected[path.name]=detail['selected']
  assert np.isfinite(pred).all();ps[arm]=pred;ps[arm+'_blend']=.9*d['ref']+.1*pred
 with (dest/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**ps)
 gate=old.gate(d,ps);r={'split':split,'scores':{k:{'rmse':rmse(d['y'],v),'fold_rmse':[rmse(d['y'][d['fid']==i],v[d['fid']==i]) for i in range(5)],'prediction_range':[float(v.min()),float(v.max())]} for k,v in ps.items()},'gate':gate,'selected':selected,'models_sha256':mh,'predictions_sha256':sha(dest/'development_predictions.npz'),'seconds':time.perf_counter()-start,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'preflight_sha256':sha(_repo_path(OUT, 'preflight.json')),'candidate_created':False};write(dest/'report.json',r);check();emit(r)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight',*SPLITS]);x=ap.parse_args();{'prepare':prepare,'preflight':preflight}.get(x.mode,lambda:run(x.mode))()
