"""Fixed nonlinear same-anchor disagreement input; linear candidates unchanged."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import disagreement_features_156 as f
import validate_base_learning_curve_v2_127 as vk
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
s=f.s;a=f.a;e=f.e;x=f.x;n=s.n;OUT=f.OUT

def width(kind):return 586 if kind.startswith('rbf') else 585
def fit(X,y,blocks,kind):
 w=width(kind);return n.fit(X[:,:w],y,blocks[:w],kind)
def predict(m,X):return n.predict(m,X[:,:width(m['kind'])])
def manual(m,X,y,V,blocks):
 w=width(m['kind']);return s.q.vn.manual(m,X[:,:w],y,V[:,:w],blocks[:w])

def prepare():
 s.check();audit=e.read(OUT/'audit.json');assert audit['passed'] and not audit['candidate_scores_computed'];assert audit['source_sha256']==e.sha(_repo_path(ROOT, 'disagreement_features_156.py'));names=['disagreement_study_156.py','disagreement_features_156.py','artifacts/disagreement-156/audit.json','artifacts/run-012/development_features.npz','reports/next-hypothesis-after154.json','reports/nested-selector-146-audit.json','reports/spatial-support-154-summary.json','reports/baseline124-weights125-summary.json','reports/candidate019-public-submission.json']
 names += [f'artifacts/bootstrap-selector-144/{split}/report.json' for split in e.SPLITS]
 r={'study':'156 same-anchor disagreement in nonlinear residual similarity','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'equivalence_audit_sha256':e.sha(OUT/'audit.json'),'one_primary':'Exactly018 except RBF1/RBF.1 use585features plus one standardized anchor-specific CNNmean3_identity-KRR scalar. none/Ridge1/Ridge.1 remain exactly585, excluding trivial linear weight-shift experiment. No alternative candidate or rescueblend.','interpretation':'RBF distance adds conditional disagreement similarity and changes train-derived median distance; not an independent measurement beyond images. New fitted prediction is not generally a fixed affine combination of branch outputs. Audit only establishes non-equivalence, not positive signal.','nested':'Outer T320/V80; A213/214 and B106/107 partitionT. A-only KRR/CNN predict BOTH B andV for feature. Anchor target onB uses A-only base and shortB with affine trained onA deepOOF. Residual innerI/H subsetB: upstream feature/anchor excludeallB; scaler/kernel/residualfit onlyI; OOF H losses choose512bootstrap frequencies withinB. Full residualfitB appliesV. Outer base affine usesT middleOOF. No V labels or global400OOF enter any fit.','small_fixed_candidates':['none(base585)','ridge1(base585)','ridge01(base585)','rbf1(joint586)','rbf01(joint586)'],'unchanged_rules':'Train-only MinMax clip/StandardScaler, each block weight1/sqrt(width), lambda*n, RBF trainmedian positive distance and centered trace normalization, same3fold seeds and512groupbootstrap as144. Extra feature is one blockwidth1. No alpha/gamma/weights/CNNseed/viewsearch.','criterion':'Run all3 existing partitions. Advance iff pooleduntrimmedMSE below018 and all independent checks pass. No fold/CI minimum, early-score stop, or posthoc variant. Ifnegative close this fixed augmentation.','validation':'All900 residualobjects independent raw scaling/linear or kernel solve; no/Ridge predictions exact018. All45 anchor feature modelIDs/stats/hashes checked; KRR independent solve and CNN CPU replay on representative block eachpartition. Independent outer/anchor affine, group/ID checks, exact018 reconstruction, outer-target poison and inner-H-target poison freshfit. Same-scope B/V feature scales are diagnostic only.','uncertainty':'10000 bootstrap draws of unique ID groups, grouping duplicate IDs by stored group key and retaining allthree partitions per ID;seed20271215. Report fixed-fit conditional95%gain CI; same400 adaptive history not independent validation.','full500_if_positive':'Lock same recipe. Reuse base/affines unchanged, A333/334 checkpoints formdelta onB166/167 and test. Refitthe downstream residualprocedure onfull500 honestblocks. Test only after fit/validation; no submission today or fromthisworker.','resources':'Existinguv CPU2/MPS.90 auditfeature nodes already extracted fromexistingcheckpoints, no neuralfits/newimages/packages/cost.900 small residualobjects plus poisonchecks.','hashes':{g:e.sha(_repo_path(ROOT, g)) for g in names},'feature_reports_sha256':audit['feature_reports_sha256'],'protected':{str(p.relative_to(ROOT)):e.sha(p) for p in [_repo_path(ROOT, 'artifacts/submission-candidate-018.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-019.csv'),_repo_path(ROOT, 'deliverables/reproduction-018-v155.zip'),_repo_path(ROOT, 'deliverables/reproduction-011-014-015-v136.zip')]},'cost':0};e.write(OUT/'plan.json',r);e.emit({'locked156':True,'plan_SHA256':e.sha(OUT/'plan.json'),'new_linear_feature_candidates':0})

def check():
 s.check();r=e.read(OUT/'plan.json')
 for name,h in {**r['hashes'],**r['feature_reports_sha256'],**r['protected']}.items():assert e.sha(_repo_path(ROOT, name))==h,name
 return r

def feature(d,split,k,j):
 path=OUT/'features'/split/f'fold-{k}-block-{j}';r=e.read(path/'report.json');z=a.readnp(path/'features.npz',r['feature_sha256']);A=z['anchor'];B=z['block'];V=z['outer'];assert set(A)|set(B)==set(np.flatnonzero(d['fid']!=k)) and set(V)==set(np.flatnonzero(d['fid']==k));assert not set(A)&set(B) and not(set(A)|set(B))&set(V);assert not set(d['g'][A])&(set(d['g'][B])|set(d['g'][V]))
 for rec in r['models']:assert e.sha(_repo_path(ROOT, rec['path']))==rec['sha256']
 np.testing.assert_array_equal(z['cnn_B'].mean(0)-z['kernel_B'],z['delta_B']);np.testing.assert_array_equal(z['cnn_V'].mean(0)-z['kernel_V'],z['delta_V']);return f.augmented(d,z),z,r

# system/verify_system below are copied from the previously audited generic residual engine,
# with fit/predict/manual dispatch selecting585 for linear and586 for nonlinear models.
def system(d,B,anchor,V,foldseed,bootseed,path):
 path.mkdir(exist_ok=False);res=d['y'][B]-anchor;folds=list(GroupKFold(3,shuffle=True,random_state=foldseed).split(d['X'][B],res,d['g'][B]));ip=np.full((5,len(B)),np.nan);hashes={}
 for t,(it,iv) in enumerate(folds):
  assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
  for ci,kind in enumerate(s.KINDS):
   model=fit(d['X'][B[it]],res[it],d['blocks'],kind);ip[ci,iv]=predict(model,d['X'][B[iv]]);file=path/f'inner-{t}-{kind}.joblib';joblib.dump({'model':model,'train_indices':B[it],'validation_indices':B[iv]},_repo_io(file));hashes[file.name]=e.sha(file)
 w,draw,wins=s.weights((ip-res)**2,d['g'][B],bootseed);pp=[]
 for kind in s.KINDS:
  model=fit(d['X'][B],res,d['blocks'],kind);pp.append(predict(model,d['X'][V]));file=path/f'full-{kind}.joblib';joblib.dump({'model':model,'train_indices':B},_repo_io(file));hashes[file.name]=e.sha(file)
 pp=np.stack(pp);primary=w@pp
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),block_indices=B,outer_indices=V,anchor=anchor,residual=res,inner_predictions=ip,weights=w,draws=draw,winners=wins,full_predictions=pp,primary=primary)
 r={'weights':dict(zip(s.KINDS,w.tolist())),'inner_MSE':dict(zip(s.KINDS,np.mean((ip-res)**2,axis=1).tolist())),'foldseed':foldseed,'bootseed':bootseed,'splits':[{'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()} for it,iv in folds],'models_sha256':hashes,'predictions_sha256':e.sha(path/'predictions.npz')};e.write(path/'report.json',r);return primary,r

def verify_system(d,B,anchor,V,path):
 r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(z['block_indices'],B);np.testing.assert_array_equal(z['outer_indices'],V);np.testing.assert_array_equal(z['anchor'],anchor);res=d['y'][B]-anchor;np.testing.assert_array_equal(z['residual'],res);ip=np.empty_like(z['inner_predictions']);maxerr=0.;count=0;folds=list(GroupKFold(3,shuffle=True,random_state=r['foldseed']).split(d['X'][B],res,d['g'][B]))
 for t,(it,iv) in enumerate(folds):
  assert r['splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
  for ci,kind in enumerate(s.KINDS):
   file=path/f'inner-{t}-{kind}.joblib';assert e.sha(file)==r['models_sha256'][file.name];obj=joblib.load(_repo_io(file));np.testing.assert_array_equal(obj['train_indices'],B[it]);np.testing.assert_array_equal(obj['validation_indices'],B[iv]);manual=manual(obj['model'],d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);actual=predict(obj['model'],d['X'][B[iv]]);err=float(abs(actual-manual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,z['inner_predictions'][ci,iv]);ip[ci,iv]=actual;count+=1
 np.testing.assert_allclose(np.mean((ip-res)**2,1),list(r['inner_MSE'].values()),atol=1e-12,rtol=0);w,wins=s.independent_weights((ip-res)**2,d['g'][B],z['draws']);np.testing.assert_array_equal(w,z['weights']);np.testing.assert_array_equal(wins,z['winners']);np.testing.assert_array_equal(w,list(r['weights'].values()));_,draw,_=s.weights((ip-res)**2,d['g'][B],r['bootseed']);np.testing.assert_array_equal(draw,z['draws']);pp=[]
 for ci,kind in enumerate(s.KINDS):
  file=path/f'full-{kind}.joblib';assert e.sha(file)==r['models_sha256'][file.name];obj=joblib.load(_repo_io(file));np.testing.assert_array_equal(obj['train_indices'],B);manual=manual(obj['model'],d['X'][B],res,d['X'][V],d['blocks']);actual=predict(obj['model'],d['X'][V]);err=float(abs(actual-manual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,z['full_predictions'][ci]);pp.append(actual);count+=1
 np.testing.assert_allclose(sum(wi*pi for wi,pi in zip(w,pp)),z['primary'],atol=1e-12,rtol=0);return z['primary'],count,maxerr

def run(split):
 check();d=f.data(split);out=OUT/split;out.mkdir(exist_ok=False);pred=np.full(400,np.nan);control=pred.copy();hashes={};start=time.perf_counter()
 for k in range(5):
  source=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(source/'report.json');fz=a.readnp(source/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];cor=[];cc=[]
  for j in range(3):
   dd,z,rec=feature(d,split,k,j);br=e.read(source/f'block-{j}/report.json');bz=a.readnp(source/f'block-{j}/predictions.npz',br['predictions_sha256']);np.testing.assert_array_equal(bz['block_indices'],z['block']);B=z['block'];path=out/f'fold-{k}-block-{j}';pp,r=system(dd,B,bz['anchor_predictions'],V,a.r.CVSEED+100*k+j,s.SEED+10000*e.SPLITS.index(split)+100*k+j,path);cor.append(pp);hashes[f'fold-{k}-block-{j}/report.json']=e.sha(path/'report.json');new=a.readnp(path/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(new['inner_predictions'][:3],bz['inner_predictions'][:3]);old=e.read(s.OUT/split/f'fold-{k}-block-{j}/report.json');oz=a.readnp(s.OUT/split/f'fold-{k}-block-{j}/predictions.npz',old['predictions_sha256']);np.testing.assert_array_equal(new['full_predictions'][:3],oz['full_candidate_predictions'][:3]);cc.append(oz['primary'])
  pred[V]=fz['base']+np.mean(cor,0);control[V]=fz['base']+np.mean(cc,0);e.emit({'split':split,'outer_complete':k})
 r0=e.read(s.OUT/split/'report.json');old=a.readnp(s.OUT/split/'predictions.npz',r0['predictions_sha256']);np.testing.assert_array_equal(control,old['primary'])
 with (out/'predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(d['ids']),groups=d['g'],y=d['y'],fold=d['fid'],reference018=control,primary=pred)
 r={'split':split,'RMSE':{'reference018':e.rmse(d['y'],control),'primary':e.rmse(d['y'],pred)},'block_reports_sha256':hashes,'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start,'linear_control_predictions_exact':True,'new_neural_fits':0};e.write(out/'report.json',r);check();e.emit(r['RMSE'])

def validate(split):
 check();d=f.data(split);out=OUT/split;rr=e.read(out/'report.json');saved=a.readnp(out/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['groups'],d['g']);np.testing.assert_array_equal(saved['y'],d['y']);count=0;errmax=0.;cpuerr=0.;kernelerr=0.
 for k in range(5):
  base_d,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);source=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(source/'report.json');fz=a.readnp(source/'predictions.npz',fr['predictions_sha256']);mb=.9*s.q.independent_affine(a.mixture(base_d['middleP'][k],x.ARM),d['y'][T],a.mixture(base_d['P'][V],x.ARM))+.1*d['short'][V];np.testing.assert_allclose(mb,fz['base'],atol=1e-9,rtol=0);cor=[]
  for j in range(3):
   dd,z,rec=feature(d,split,k,j);A,B,M,short,_=base_d['cells'][k,j];np.testing.assert_array_equal(A,z['anchor']);np.testing.assert_array_equal(B,z['block']);path=out/f'fold-{k}-block-{j}';br=e.read(path/'report.json');assert e.sha(path/'report.json')==rr['block_reports_sha256'][f'fold-{k}-block-{j}/report.json'];bz=a.readnp(path/'predictions.npz',br['predictions_sha256']);anchor=.9*s.q.independent_affine(a.mixture(base_d['deepP'][k,j],x.ARM),d['y'][A],a.mixture(M,x.ARM))+.1*short;np.testing.assert_allclose(anchor,bz['anchor'],atol=1e-9,rtol=0);pp,c,err=verify_system(dd,B,bz['anchor'],V,path);count+=c;errmax=max(errmax,err);cor.append(pp)
   if k==j==0:
    km=joblib.load(_repo_path(ROOT, rec['models'][0]['path']))['model']
    for query,key in [(B,'B'),(V,'V')]:
     kp=vk.manual_kernel(km,d['K'][A],d['y'][A],d['K'][query]);ke=float(abs(kp-z[f'kernel_{key}']).max());assert ke<1e-8;kernelerr=max(kernelerr,ke)
     for si,item in enumerate(rec['models'][1:]):
      obj=torch.load(_repo_path(ROOT, item['path']),weights_only=True,map_location='cpu');pp=x.predict(obj,d,query,'cpu');ce=float(abs(pp-z[f'cnn_{key}'][si]).max());assert ce<1e-3;cpuerr=max(cpuerr,ce)
    pert=dict(dd);pert['y']=d['y'].copy();pert['y'][V]+=100000;pp,_=system(pert,B,bz['anchor'],V,br['foldseed'],br['bootseed'],OUT/f'poison-{split}');np.testing.assert_array_equal(pp,bz['primary']);I=np.array(br['splits'][0]['train_indices']);H=np.array(br['splits'][0]['validation_indices']);target=d['y'].copy();target[H]+=54321;lookup={int(ii):ix for ix,ii in enumerate(B)};rI=target[I]-bz['anchor'][[lookup[int(ii)] for ii in I]]
    for kind in s.KINDS:
     m=fit(dd['X'][I],rI,dd['blocks'],kind);old=joblib.load(_repo_io(path/f'inner-0-{kind}.joblib'))['model'];np.testing.assert_array_equal(predict(m,dd['X'][H]),predict(old,dd['X'][H]))
  np.testing.assert_array_equal(fz['base']+np.mean(cor,0),saved['primary'][V])
 r={'passed':True,'split':split,'models_verified':count,'independent_max_error':errmax,'CPU_MPS_feature_max_error':cpuerr,'independent_upstream_kernel_max_error':kernelerr,'outer_target_poison_exact':True,'inner_H_target_poison_5_fits_exact':True,'all45_feature_train_scope_audit':e.sha(OUT/'audit.json'),'upstream_training_exclusion_certificate':e.sha(_repo_path(ROOT, 'reports/nested-selector-146-audit.json')),'report_sha256':e.sha(out/'report.json')};e.write(OUT/f'validation-{split}.json',r);check();e.emit(r)

def summary():
 check();E={key:[] for key in ['reference018','primary']};scores={};ids=None;groups=None
 for split in e.SPLITS:
  r=e.read(OUT/split/'report.json');v=e.read(OUT/f'validation-{split}.json');assert v['passed'] and v['report_sha256']==e.sha(OUT/split/'report.json');z=a.readnp(OUT/split/'predictions.npz',r['predictions_sha256']);scores[split]=r['RMSE']
  if ids is None:ids=z['ids'];groups=z['groups']
  else:np.testing.assert_array_equal(ids,z['ids']);np.testing.assert_array_equal(groups,z['groups'])
  for key in E:E[key].append((z[key]-z['y'])**2)
 E={k:np.stack(v) for k,v in E.items()};M={k:float(v.mean()) for k,v in E.items()};unique=np.unique(groups);gi=[np.flatnonzero(groups==g) for g in unique];rng=np.random.default_rng(20271215);gains=[]
 for _ in range(10000):
  ii=np.concatenate([gi[i] for i in rng.integers(0,len(gi),len(gi))]);gains.append(float(np.sqrt(E['reference018'][:,ii].mean())-np.sqrt(E['primary'][:,ii].mean())))
 r={'study':'156 nonlinear same-anchor disagreement','advance':M['primary']<M['reference018'],'validation_passed':True,'RMSE_by_partition':scores,'pooled_MSE':M,'pooled_RMSE':{k:float(np.sqrt(v)) for k,v in M.items()},'conditional_ID_group_gain_CI95':np.quantile(gains,[.025,.975]).tolist(),'unique_IDs':len(ids),'unique_groups':len(unique),'same400_adaptive_history':True,'linear_candidates_unchanged':True,'current_best':'018','test_or_holdout_read':False,'new_neural_fits':0,'cost':0,'plan_sha256':e.sha(OUT/'plan.json')};e.write(OUT/'decision.json',r);e.write(_repo_path(ROOT, 'reports/disagreement-156-summary.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','summary',*e.SPLITS,*['validate-'+v for v in e.SPLITS]]);v=p.parse_args().mode
 if v=='prepare':prepare()
 elif v=='summary':summary()
 elif v.startswith('validate-'):validate(v[9:])
 else:run(v)
