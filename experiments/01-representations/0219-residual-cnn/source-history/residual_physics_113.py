"""Honest-block residual regression on fixed57 morphology features; no globalOOF fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,datetime,argparse,importlib.metadata,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==ROOT/'.venv'
import numpy as np,joblib
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import nested_fusion_112 as prior
sha=prior.sha;read=prior.read;write=prior.write;emit=prior.emit;rmse=prior.rmse
OUT=ROOT/'artifacts/residual-physics-113';SPLITS=prior.SPLITS;CANDIDATES=('none','ridge1','ridge01');LAMBDAS={'ridge1':1.,'ridge01':.1};CVSEED=20271013

def load(split):
 path=prior.OUT/split/'predictions.npz';r=read(prior.OUT/split/'report.json');v=read(prior.OUT/f'validation-{split}.json');assert v['passed'] and sha(prior.OUT/split/'report.json')==v['report_sha256'] and sha(path)==r['prediction_sha256']
 with np.load(path,allow_pickle=False) as z:ids=z['ids'].tolist();y=z['y'].copy();fid=z['fold'].copy();ref=z['reference011'].copy()
 manifest=read(ROOT/'artifacts/run-001/split_manifest.json');assert ids==sorted(manifest['development_folds']) and len(ids)==400 and not set(ids)&set(manifest['holdout']);groups=np.array([manifest['group_by_id'][i] for i in ids]);panel=read(ROOT/'artifacts/run-022/report.json')
 with np.load(ROOT/'artifacts/run-022/development_features.npz',allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['groups'],groups);X=z['X'].astype(float)
 assert X.shape==(400,57) and np.isfinite(X).all();assert len(panel['blocks'])==57
 return {'ids':ids,'y':y,'fid':fid,'reference011':ref,'g':groups,'X':X,'blocks':panel['blocks']}

def anchor(d,split,k,j,verify_files=False):
 tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);f=prior.OUT/split/f'fold-{k}';fr=read(f/'report.json');cell=f/f'inner-{j}';cr=read(cell/'report.json');assert sha(cell/'report.json')==fr['cell_report_sha256'][f'inner-{j}/report.json'] and sha(cell/'calibration.npz')==cr['calibration_sha256']
 with np.load(cell/'calibration.npz',allow_pickle=False) as z:a=z['train_indices'].copy();b=z['validation_indices'].copy();p=z['p011'].copy();p009=z['p009'].copy();small=z['shortB'].copy();nested=z['inner_oof008'].copy();p008=z['p008'].copy()
 assert set(a)|set(b)==set(tr) and not set(a)&set(b) and not (set(a)|set(b))&set(va);assert not set(d['g'][a])&set(d['g'][b]) and not set(d['g'][tr])&set(d['g'][va])
 np.testing.assert_array_equal(p,.9*p009+.1*small);center=nested.mean();coef=np.linalg.lstsq(np.column_stack([np.ones(len(a)),nested-center]),d['y'][a],rcond=None)[0];np.testing.assert_allclose(p009,coef[0]+coef[1]*(p008-center),atol=1e-10,rtol=0)
 seen=[]
 for ell in range(3):
  deep=cell/f'deep-{ell}';dr=read(deep/'report.json');assert sha(deep/'report.json')==cr['deep'][ell]['report_sha256'];dt=set(dr['train_indices']);dv=set(dr['validation_indices']);assert dt|dv==set(a) and not dt&dv and not (dt|dv)&(set(b)|set(va));assert set(dr['outer_validation_indices'])==set(range(400))-set(a);seen.extend(dr['validation_indices'])
  if verify_files:
   for n,h in dr['files_sha256'].items():assert sha(deep/n)==h,n
 assert sorted(seen)==sorted(a.tolist())
 old=prior.cal.OUT/split/f'fold-{k}'/f'inner-{j}';orec=read(old/'report.json');assert orec['train_indices']==a.tolist() and orec['validation_indices']==b.tolist()
 if verify_files:
  for n,h in orec['files_sha256'].items():assert sha(old/n)==h,n
 sr=read(cell/'shortB/report.json');assert sha(cell/'shortB/report.json')==cr['model_reports']['shortB'];assert sha(cell/'shortB/predictions.npz')==sr['predictions_sha256']
 if verify_files:assert sha(cell/'shortB/model.pt')==sr['model_sha256']
 return a,b,p,{'calibration_sha256':cr['calibration_sha256'],'cell_report_sha256':sha(cell/'report.json'),'anchor_training_indices':a.tolist(),'residual_block_indices':b.tolist(),'outer_validation_indices':va.tolist(),'deep_anchor_models_exclude_residual_block_and_outer':True}

def prepare():
 prior.check();OUT.mkdir(exist_ok=False);assert read(prior.OUT/'validation-replay.json')['passed'] and read(prior.OUT/'incumbent-head-control-validation.json')['passed'];start=time.perf_counter();verification=[];inputs={}
 for split in SPLITS:
  d=load(split)
  for k in range(5):
   for j in range(3):
    a,b,p,v=anchor(d,split,k,j,True);verification.append({'split':split,'fold':k,'block':j,'anchor_rows':len(a),'residual_rows':len(b),**v})
  for p in [prior.OUT/split/'report.json',prior.OUT/split/'predictions.npz',prior.OUT/f'validation-{split}.json']:inputs[str(p.relative_to(ROOT))]=sha(p)
 for p in [ROOT/'artifacts/run-022/development_features.npz',ROOT/'artifacts/run-022/report.json',ROOT/'reports/nested-fusion-112-summary.json',prior.OUT/'validation-replay.json',ROOT/'reports/contrast-calibration-077-validation.json',ROOT/'joint_experiment_047.py',ROOT/'residual_diagnostics_076.py',ROOT/'cnn_residual.py']:inputs[str(p.relative_to(ROOT))]=sha(p)
 write(OUT/'anchor-audit.json',{'passed':True,'blocks':verification,'seconds':time.perf_counter()-start,'scope':'112upstreammodels strictlyexclude eachresidualblockandoutervalidation; verifiedhashes+indices+independentOLS. No newupstreamfit.'})
 p={'study':'113 honest-block physics57 residual regression','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Fixedmorphology57 mayexplain signed011 errors evenwhenitsdirecttargetpredictionisnotcomplementary. No newfeature,segmentation,augmentation orPublictuning.',
 'duplicate_audit':{'022_101':'Directtarget KRR onphysics57;101fixed10% blend. Not residualtargetregression.','047':'JointDINO/morphology targetkernel;explicitlynoresidualtargets.','074_077':'Affinepredictioncalibration andcomponentcontrast,notgeometry57-conditionedresiduallearner.','076':'Residualcorrelationdiagnostics,nottrainedphysicsresidualmodel.','cnn_residual':'Residualnetworkarchitecture,notresiduallabellearning.','112':'Nested011 versuspoolingpredictionmixtures;providesverifiedanchorcache,notphysicsresidualmodel.'},
 'honest_block_design':'Foroutertrain320 andoutervalidation80, reuse3anchors trainedonA_j~213 thatexcludeB_j~107, withentire011calibration nestedinsideA_j. Residualblocktarget r_j=y[B_j]-p011_A_j(B_j). WithinB_j ONLY, GroupKFold3 selectnone/Ridge1/Ridge.1;allresidualfeature/targetstatisticsfit B_j_innertrain~71,notB_j_validation~36. AnchortrainedA_j excludesALL B_j; hencealsoexcludesresidualinnervalidation. Eachinnertraininginformation setisA_j union B_j_innertrain. No naiveCVofpooled320crossfitresiduals, whoseotheranchorscouldhaveseeninner validationlabels.',
 'model':'Existingfixed57 columns andblocklabels. MinMaxScaler(clip=True) thenStandardScaler fittedonlyresidualtrain; multiplyeachcolumnby1/sqrt(numberofcolumnsinitspredefinedblock). Residualscale=RMS(trainingresidual),fallback1 if<=1e-12;no residualcentering. Ridge(fit_intercept=False,solver=cholesky),alpha=lambda*n_residualtrain. Featurecentering implieszeromeantrainingcorrection;nofreeinterceptornewglobalaffinecalibration.',
 'candidates':{'order_for_ties':list(CANDIDATES),'lambdas':LAMBDAS,'none':'identicallyzero,nofit','inner_selector':'MinimumpooleduntrimmedresidualRMSE across3innerfolds; exacttiesprefernone thenstrongerridge1 thenridge.1. No outerlabels oraffected-row exclusion.'},
 'primary':'011_outer320(X_outer80) + mean_j=0..2 g_j(X_outer80). Eachg_j isselectedandrefittedonlyonitswholehonestB_j(~107). Uniformthree-waymean andfullresidualcorrection;no10% blend,globalOOFweightselection orposthocamplitudetuning. Allthreeblockmodelsretained,includingzerochoices.',
 'training_size_limit':'Residualmodelseachsee~107 ratherthanpooled320;blockvarianceisatradeoffforhonestcheapreuse. Residualtargetsarefrom213-trainedanchorswhilepredictionanchoristrained320,whichmaychangeerrorstructure. ThisispartofthefixedalgorithmassessedbyouterOOF,notclaimsoftransportinvariance.',
 'assessment':'Runallthreeexistingpartitionsunchanged. PrimarypooledRMSE onall400 rows;011unchangedcontrol. Foldsigns,conditionalpairedCIandfixed092top5exclusiondiagnosticonly. Considercandidateonlyifprimarystrictlyimproves011inallthreeandindependentvalidationpasses. Same400historicalselectionbiaspersists;resplitsnotindependent.',
 'verification':'Traceeveryanchortrain/validation index andsourcehash; independentmanual feature transform anddualRidge; recomputeeveryinnercandidate/selection,joblibreload,outerlabelperturbation fullresidualfoldreplay,innerheldoutlabelperturbation fits. Reuse112fullupstreamlabel-perturbation certificate,whoseperturbed187 rowsinclude107residualblock+80outervalidation.',
 'resources':'CPU2existinguv,270innerRidgefits+upto45selectedrefits,nomorebackbonetraining. Firstblocktimed,reused;forecast45blocks<120s,RSS<2GB. Existingcachedfeaturesonly;no newrawimages,weights,packages,payment. Verificationexpectedseconds–minutes.',
 'if_pass':'Lockthisprocedure,thenfull500finaltrainingonly. Toobtain3honestresidualblocks(~167),fit3anchor011models oncomplements(~333),includingnestedcalibrationwithin333;choose/refitresidualmodelswithsamewithin-blockCV. Applyaveragecorrectiontoalreadyfull500011. Test1000onlyafterlockandfit. VerifyCSV/reload/IDs/order/finite/SHA;parentcoordinatesupload.',
 'scope':'No newholdout/test selection. Preserve011andallfailures. No newcandidateCSVunlessconsistentvalidatedimprovement.','anchor_audit_sha256':sha(OUT/'anchor-audit.json'),'sources':{n:sha(ROOT/n) for n in ['residual_physics_113.py','validate_residual_physics_113.py','nested_fusion_112.py']},'inputs':inputs,'protected011':read(ROOT/'reports/candidate011-public-submission.json')['protected_sha256'],'environment':{d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}}
 write(OUT/'plan.json',p);emit({'prepared':True,'45_anchors_verified':True,'audit_seconds':time.perf_counter()-start,'plan_sha256':sha(OUT/'plan.json')})

def check():
 p=read(OUT/'plan.json')
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(ROOT/n)==h,n
 assert sha(OUT/'anchor-audit.json')==p['anchor_audit_sha256'];return p

def fit_model(X,r,blocks,kind):
 if kind=='none':return {'kind':'none','n_train':len(r)}
 bound=MinMaxScaler(clip=True).fit(X);B=bound.transform(X);scale=StandardScaler().fit(B);w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);Z=scale.transform(B)*w;rs=float(np.sqrt(np.mean(r*r)));rs=rs if rs>1e-12 else 1.;alpha=LAMBDAS[kind]*len(r);reg=Ridge(alpha=alpha,fit_intercept=False,solver='cholesky').fit(Z,r/rs)
 return {'kind':kind,'bound':bound,'scale':scale,'weights':w,'residual_scale':rs,'alpha':alpha,'regressor':reg,'n_train':len(r)}
def predict(m,X):
 if m['kind']=='none':return np.zeros(len(X))
 return m['regressor'].predict(m['scale'].transform(m['bound'].transform(X))*m['weights'])*m['residual_scale']

def fit_block(d,B,panchor,k,j,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();X=d['X'];r=d['y'][B]-panchor;groups=d['g'][B];folds=list(GroupKFold(3,shuffle=True,random_state=CVSEED+100*k+j).split(X[B],r,groups));ip=np.full((3,len(B)),np.nan);models={};splits=[]
 for t,(tr,va) in enumerate(folds):
  assert not set(groups[tr])&set(groups[va]);splits.append({'train_indices':B[tr].tolist(),'validation_indices':B[va].tolist()})
  for ci,kind in enumerate(CANDIDATES):
   m=fit_model(X[B[tr]],r[tr],d['blocks'],kind);ip[ci,va]=predict(m,X[B[va]]);f=path/f'inner-{t}-{kind}.joblib';joblib.dump({'model':m,'train_indices':B[tr],'validation_indices':B[va]},f);models[f.name]=sha(f)
 scores=[rmse(r,p) for p in ip];chosen=CANDIDATES[int(np.argmin(scores))];m=fit_model(X[B],r,d['blocks'],chosen);joblib.dump({'model':m,'train_indices':B,'train_ids':[d['ids'][i] for i in B],'chosen':chosen},path/'selected.joblib')
 outer=np.flatnonzero(d['fid']==k);correction=predict(m,X[outer]);assert np.isfinite(correction).all()
 with (path/'fit_data.npz').open('xb') as f:np.savez_compressed(f,block_indices=B,anchor_predictions=panchor,residual=r,inner_predictions=ip,outer_indices=outer,correction=correction)
 rec={'outer':k,'block':j,'n_train':len(B),'chosen':chosen,'inner_scores':dict(zip(CANDIDATES,scores)),'inner_splits':splits,'models_sha256':models,'selected_sha256':sha(path/'selected.joblib'),'fit_data_sha256':sha(path/'fit_data.npz'),'seconds':time.perf_counter()-start};write(path/'report.json',rec);return correction,rec

def run(split):
 check();d=load(split);out=OUT/split;out.mkdir(exist_ok=False);start=time.perf_counter();c=np.full((3,400),np.nan);records=[]
 for k in range(5):
  fd=out/f'fold-{k}';fd.mkdir();va=np.flatnonzero(d['fid']==k);blocks=[];covered=[]
  for j in range(3):
   A,B,pa,trace=anchor(d,split,k,j);pp,rec=fit_block(d,B,pa,k,j,fd/f'block-{j}');c[j,va]=pp;covered.extend(B.tolist());blocks.append({'block':j,'chosen':rec['chosen'],'report_sha256':sha(fd/f'block-{j}/report.json'),'anchor_trace':trace})
   if split=='original' and k==0 and j==0:
    b={'seconds_first_block_reused':rec['seconds'],'projected45blocks_seconds':45*rec['seconds'],'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};b['resource_pass']=b['projected45blocks_seconds']<120 and b['rss_bytes']<2e9;write(OUT/'benchmark.json',b);assert b['resource_pass']
  assert sorted(covered)==np.flatnonzero(d['fid']!=k).tolist();r={'fold':k,'blocks':blocks};write(fd/'report.json',r);records.append(r);emit({'split':split,'completed_outer':k,'selected':[v['chosen'] for v in blocks],'seconds':time.perf_counter()-start})
 correction=c.mean(0);primary=d['reference011']+correction;np.testing.assert_array_equal(d['reference011']+np.zeros(400),d['reference011']);ps={'reference011':d['reference011'],'primary':primary};scores={n:{'rmse':rmse(d['y'],p),'fold_rmse':[rmse(d['y'][d['fid']==k],p[d['fid']==k]) for k in range(5)]} for n,p in ps.items()};gain=scores['reference011']['rmse']-scores['primary']['rmse'];ix=np.random.default_rng(20271014).integers(0,400,(3000,400));rr=(d['y']-ps['reference011'])**2;nn=(d['y']-primary)**2;ci=np.quantile(np.sqrt(rr[ix].mean(1))-np.sqrt(nn[ix].mean(1)),[.025,.975]).tolist();keep=np.array([i not in prior.FIXED5 for i in d['ids']])
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(f,ids=np.array(d['ids']),y=d['y'],fold=d['fid'],block_corrections=c,correction=correction,**ps)
 r={'split':split,'scores':scores,'RMSE_gain':gain,'fold_RMSE_gains':(np.array(scores['reference011']['fold_rmse'])-np.array(scores['primary']['fold_rmse'])).tolist(),'conditional95_RMSE_gain_CI':ci,'fixed092_other395_gain_diagnostic':rmse(d['y'][keep],ps['reference011'][keep])-rmse(d['y'][keep],primary[keep]),'chosen_counts':{n:sum(b['chosen']==n for r in records for b in r['blocks']) for n in CANDIDATES},'correction_RMS':float(np.sqrt(np.mean(correction**2))),'fold_report_sha256':{f'fold-{k}/report.json':sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};write(out/'report.json',r);check();emit(r)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare',*SPLITS]);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
