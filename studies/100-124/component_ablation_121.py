"""Four fixed leave-group-out diagnostics, with train-local downstream reconstruction."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,csv,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import ranking_objective_120 as previous
import nonlinear_residual_115 as n
import numpy as np,joblib,torch
from sklearn.model_selection import GroupKFold
e=n.e;r=n.r;OUT=_repo_path(ROOT, 'artifacts/component-ablation-121')
COLS=('baseline','kernel','cnn','Smean','Bmean');W=np.array([.375,.1875,.1875,.125,.125]);ARMS=('no_DINO','no_CNN','no_legacy_physics');MASKS={'full':np.ones(5,bool),'no_DINO':np.array([1,1,1,0,0],bool),'no_CNN':np.array([1,1,0,1,1],bool),'no_legacy_physics':np.array([0,0,1,1,1],bool)}
SOURCES={}
def tracked(p,expected=None):
 h=e.sha(p)
 if expected is not None:assert h==expected,str(p)
 SOURCES[str(p.relative_to(ROOT))]=h;return p

def readcsv(p):
 with tracked(p).open() as f:return list(csv.DictReader(f))
def readnp(p,h=None):
 with np.load(_repo_io(tracked(p,h)),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def mixture(P,arm):
 w=W*MASKS[arm];return P@(w/w.sum())
def affine(trainpred,y,pred):
 center=float(trainpred.mean());target=float(y.mean());z=trainpred-center;slope=float(z@(y-target)/(z@z));return target+slope*(pred-center),{'prediction_center':center,'target_center':target,'slope':slope,'intercept':target-slope*center}
def assemble(p009,short,arm):return p009 if arm=='no_DINO' else .9*p009+.1*short

def bundle(path):
 rec=e.read(path/'report.json');tracked(path/'report.json');z=readnp(path/'predictions.npz',rec['files_sha256']['predictions.npz']);P=np.column_stack([z[k] for k in COLS]);np.testing.assert_allclose(mixture(P,'full'),z['ensemble008'],atol=1e-12,rtol=0);assert z['train_indices'].tolist()==rec['train_indices'] and z['validation_indices'].tolist()==rec['validation_indices'];return z['train_indices'],z['validation_indices'],P

def load(split,audit=False):
 d=r.load(split);ids=d['ids'];y=d['y'];fid=d['fid'];track=_repo_path(ROOT, 'artifacts/dinob-complement-064', split, 'development_predictions.npz');v=e.read(_repo_path(ROOT, 'reports/dinob-complement-064-validation.json'))['results'][split];z=readnp(track,v['predictions_sha256']);assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['fold'],fid)
 d['reference008']=z['mixed50'];d['P']=np.column_stack([np.zeros((400,3)),z['Smean'],z['Bmean']]);p003=z['reference003']
 sv=e.read(_repo_path(ROOT, 'reports/frozen-complement-085-validation.json'))['results'][split];f=readnp(_repo_path(ROOT, 'artifacts/frozen-complement-085', split, 'development_predictions.npz'),sv['predictions_sha256']);assert f['ids'].tolist()==ids;np.testing.assert_array_equal(f['primary'],d['reference011']);d['short']=f['frozen'];d['reference009']=f['reference009']
 nr=e.read(n.OUT/split/'report.json');z=readnp(n.OUT/split/'predictions.npz',nr['predictions_sha256']);d['reference013']=z['primary'];d['correction']=z['corrections'].mean(0);np.testing.assert_array_equal(d['reference011']+d['correction'],d['reference013'])
 z=readnp(e.cal.OUT/'development_features.npz',e.read(e.cal.OUT/'extraction.json')['feature_sha256']);assert z['ids'].tolist()==ids;Xbase=z['Xbase'];images=z['images'];z=readnp(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'));assert z['ids'].tolist()==ids;K=z['X'];errors=[];modelinfo=None
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);path=_repo_path(ROOT, 'artifacts/dino-frozen-037/regression', split, f'fold-{k}');rec=e.read(path.parent/f'fold-{k}-report.json');tracked(path.parent/f'fold-{k}-report.json');b=joblib.load(_repo_io(tracked(path/'baseline.joblib',rec['sha256']['baseline.joblib'])));q=joblib.load(_repo_io(tracked(path/'compact-kernel.joblib',rec['sha256']['compact-kernel.joblib'])));assert b['train_ids']==q['train_ids']==[ids[i] for i in tr];assert b['validation_ids']==q['validation_ids']==[ids[i] for i in va];bp=sum(b['weights'][name]*model.predict(Xbase[va]) for name,model in b['models'].items());kp=q['model'].predict(K[va]);cp=(p003[va]-.5*bp-.25*kp)/.25;d['P'][va,:3]=np.column_stack([bp,kp,cp]);modelinfo={'weights':b['weights'],'models':{name:str(model) for name,model in b['models'].items()}}
  if audit:
   a=torch.load(_repo_io(tracked(path/'tiny128.pt',rec['sha256']['tiny128.pt'])),map_location='cpu',weights_only=True);assert a['train_ids']==[ids[i] for i in tr] and a['validation_ids']==[ids[i] for i in va];m=e.cal.fv.engine.TinyCNN();m.load_state_dict(a['state_dict']);m.eval();norm=a['normalization'];pred=[]
   with torch.inference_mode():
    for st in range(0,len(va),16):pred.extend((m(torch.tensor((images[va[st:st+16]]-norm['x_mean'])/max(norm['x_std'],1e-6)))).numpy().astype(float))
   pred=np.array(pred)*norm['y_std']+norm['y_mean'];err=float(abs(pred-cp).max());assert err<1e-3;errors.append(err)
 np.testing.assert_allclose(mixture(d['P'],'full'),d['reference008'],atol=1e-12,rtol=0)
 d['cells']={};d['middleP']={};d['trains']={};d['inners']={};d['deepP']={};d['deep_short']={}
 for k in range(5):
  T=np.flatnonzero(fid!=k);V=np.flatnonzero(fid==k);mid=np.full((len(T),5),np.nan);lookup={int(i):j for j,i in enumerate(T)}
  for j in range(3):
   A,B,old,trace=r.anchor(d,split,k,j);AA,BB,P=bundle(e.cal.OUT/split/f'fold-{k}/inner-{j}');np.testing.assert_array_equal(A,AA);np.testing.assert_array_equal(B,BB);mid[[lookup[int(i)] for i in B]]=P;cell=e.OUT/split/f'fold-{k}/inner-{j}';cr=e.read(cell/'report.json');z=readnp(cell/'calibration.npz',cr['calibration_sha256']);d['cells'][k,j]=(A,B,P,z['shortB'],old);deep=np.full((len(A),5),np.nan);lookupA={int(i):a for a,i in enumerate(A)};seen=[]
   for ell in range(3):
    a,b,dp=bundle(cell/f'deep-{ell}');assert set(a)|set(b)==set(A) and not set(a)&set(b) and not (set(a)|set(b))&(set(B)|set(V));deep[[lookupA[int(i)] for i in b]]=dp;seen.extend(b.tolist())
   assert sorted(seen)==sorted(A.tolist());d['deepP'][k,j]=deep
   aa,coef=affine(mixture(deep,'full'),y[A],mixture(P,'full'));np.testing.assert_allclose(assemble(aa,z['shortB'],'full'),old,atol=1e-10,rtol=0)
  d['middleP'][k]=mid
  aa,coef=affine(mixture(mid,'full'),y[T],mixture(d['P'][V],'full'));np.testing.assert_allclose(aa,d['reference009'][V],atol=1e-10,rtol=0);np.testing.assert_allclose(assemble(aa,d['short'][V],'full'),d['reference011'][V],atol=1e-10,rtol=0)
 d['audit']={'CNN15_reload_max_error_this_split':max(errors) if errors else None,'legacy_baseline':modelinfo,'exact013_formula':True,'no_raw_image_reads':True};return d

def prepare():
 previous.check();OUT.mkdir(exist_ok=False);torch.set_num_threads(2);audits={}
 for split in e.SPLITS:audits[split]=load(split,True)['audit'];e.emit({'source_audit':split,'passed':True})
 lineage={'008':'.375 baseline795 + .1875 compact26KRR + .1875 TinyCNN128 + .125 DINO-S mean3 + .125 DINO-B mean3','009':'Foldtrain-only affine of008 from3innerOOF fits;320train uses213/214 calibration sources','011':'.9 calibrated009 + .1 DINO-B shorthead32/12epochs','013':'011 + mean3 selected physics57 residualmodels; eachtrainedon106/107 honestrows using213/214 whole011 anchors','legacy_physics':'baseline795 combinesbase_et .375,base_ridge .375,morph_ridge .25; morphology/texture/statisticalproxies,notexactphysicalequation. Compact26 containsinverse_sqrt_grain_diameter,phase/porefractions,alignment,lineintercepts etc.','new_residual':'57features=compact26+TV31, none/Ridge1/Ridge.1/RBF1/RBF.1 choseninsidehonestblock. Distinctfromlegacydirecttargetregression.','DINO':'FrozenS/B full504 multilayer6/9/12CLS+patchmean/std andMLP. No supervisedbackbonefinetuningin013.','CNN':'Tiny128 BOX,80epochs,D4train,identityinference,singleseed,last5weightaverage.','group_scope':'no_legacy_physics removesonlylegacydirecttargetregressors;57physicalresidualfeatures remain bydesign. no_DINO removesbothlongheadsandshorthead throughoutanchorsandtop-level. no_CNN removesCNNthroughoutanchorsandtop-level.','domain_limit':'Proxyfeaturesandimagebasedlearnersdo notproveactualgrainmasks,absolutephysicalscale orcausalhardnesslaw.'}
 plan={'study':'121 fixed4componentablation','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'lineage':lineage,'four_groups':['DINO','CNN','legacy_physics','new_residual'],'full_pipeline_arms':list(ARMS)+['no_new_residual'],'primary_estimand':'Leaveonegroupout procedure. Forfirst3arms,dropappropriate008columnsandrenormalizeremainingoriginalweights to1. no_DINOalsoomits011shortheadanduses009weight1;othersretaintop.9/.1. Refit009OLSusingoldinnercomponentOOFONLY,includingdeepOOFforresidualanchors;refit115residualselectorandmodelsagainstnewanchors usingunchanged57features/CVseed/candidates. no_new_residual exactly011.','frozen_diagnostic':'Separatefoldtrainmeanreplacement: current013minus .9*slope*sum(weight*(component-mu));DINOalsominus.1*(short-mu). Retainoriginalcalibrationandresidual. Newresidualsubtractscorrection. Thisiscurrentpipelineoutputintervention,notretrainedfeatureimportance. No rawzerohardness/headweights.','reuse_vs_refit':'Existingbasepredictorstrainedindependentlyontheexactsamefolds anddonotconsumeotherbranches;theircheckpoints/cachecanbereusedafterbranchremoval. Affinecalibrators andresidualtargets/models dependonmixture and MUST berefit;donehere. Removinginputfeaturesinsidebaseline/KRR orsharedneuralnetwouldrequireitsownretraining andisnotdone. No changingremainingarchitectures/hyperparameters/trainingbudget.','assessment':'All3existingpartitions pooleduntrimmedMSE;reportall4 nochoosingfavorablefolds/IDs. ConditionalIDclusterbootstrap10000seed20271027. Standalonepredictionerrorsanderrorcorrelationsdiagnosticonly;correlationdoesnotproveabsenceofincrementalinformation. No bestsubset/weightfit ornewcandidateCSV fromthisdiagnostictask.','validation':'Reconstruct013andall008/009/011anchors;reloadouterbaseline/KRR/CNN. Verifycachehashandtrainindices; independentlyOLSandrawkernel solveeverynewfit; outerlabelperturbationrepeatall3removedgroups onefold andinnerheldoutlabelcheck. Original112upstreamtraininglabelinterventioncertificate reused.','resource':'ExistingCPU2,threearms*45residualblocks;2160serializedfitsincludingnone;no neuralfits/images/externaldata/test/holdout. Firstblockproject135<180s,RSS<2GB.','source_sha256':{name:e.sha(_repo_path(ROOT, name)) for name in ['component_ablation_121.py','validate_component_ablation_121.py']},'inputs_sha256':dict(SOURCES),'audits':audits,'protected':{**previous.check()['protected'],**{name:e.sha(_repo_path(ROOT, name)) for name in ['reports/ranking-objective-120-summary.json','reports/ranking-objective-120-preservation.json']}},'environment':previous.check()['environment'],'candidate_policy':'Preserve013 regardlessofdiagnosticranking;proposeonejustifiedfollowupafterallresults. No automaticfull500/test/CSV/upload.'};e.write(_repo_path(OUT, 'plan.json'),plan);e.emit({'prepared':True,'source_files':len(SOURCES),'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 previous.check();p=e.read(_repo_path(OUT, 'plan.json'))
 for name,h in {**p['source_sha256'],**p['inputs_sha256'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, name))==h,name
 return p

def fit_block(d,B,anchor,k,j,path):
 path.mkdir(exist_ok=False);res=d['y'][B]-anchor;V=np.flatnonzero(d['fid']==k);ip=np.full((5,len(B)),np.nan);folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]));hashes={};start=time.perf_counter()
 for t,(it,iv) in enumerate(folds):
  assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
  for ci,kind in enumerate(n.KINDS):
   m=n.fit(d['X'][B[it]],res[it],d['blocks'],kind);ip[ci,iv]=n.predict(m,d['X'][B[iv]]);file=path/f'inner-{t}-{kind}.joblib';joblib.dump({'model':m,'train_indices':B[it],'validation_indices':B[iv]},_repo_io(file));hashes[file.name]=e.sha(file)
 scores=np.mean((ip-res[None,:])**2,1);chosen=n.KINDS[int(scores.argmin())];m=n.fit(d['X'][B],res,d['blocks'],chosen);cor=n.predict(m,d['X'][V]);joblib.dump({'model':m,'train_indices':B},_repo_io(path/'selected.joblib'));hashes['selected.joblib']=e.sha(path/'selected.joblib')
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),block_indices=B,outer_indices=V,anchor_predictions=anchor,residual=res,inner_predictions=ip,correction=cor)
 if path==_repo_path(OUT, 'original/no_DINO-fold-0/block-0'):
  import resource
  b={'first_block_seconds_reused':time.perf_counter()-start,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};b['projected135blocks_seconds']=135*b['first_block_seconds_reused'];b['passed']=b['projected135blocks_seconds']<180 and b['RSS_bytes']<2e9;e.write(_repo_path(OUT, 'benchmark.json'),b);assert b['passed']
 rec={'chosen':chosen,'inner_MSE':dict(zip(n.KINDS,scores.tolist())),'splits':[{'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()} for it,iv in folds],'models_sha256':hashes,'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(path/'report.json',rec);return cor,rec

def fold(d,k,arm,path):
 path.mkdir(exist_ok=False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);p009,cal=affine(mixture(d['middleP'][k],arm),d['y'][T],mixture(d['P'][V],arm));base=assemble(p009,d['short'][V],arm);cor=[];anchors=[];coef=[];records=[]
 for j in range(3):
  A,B,P,short,_=d['cells'][k,j];pp,c=affine(mixture(d['deepP'][k,j],arm),d['y'][A],mixture(P,arm));anchor=assemble(pp,short,arm);cc,rec=fit_block(d,B,anchor,k,j,path/f'block-{j}');cor.append(cc);anchors.append(anchor);coef.append(c);records.append(rec)
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),outer_indices=V,base=base,corrections=np.stack(cor),primary=base+np.mean(cor,0))
 rec={'arm':arm,'fold':k,'outer_calibrator':cal,'anchor_calibrators':coef,'selected':[z['chosen'] for z in records],'block_reports_sha256':{f'block-{j}/report.json':e.sha(path/f'block-{j}/report.json') for j in range(3)},'predictions_sha256':e.sha(path/'predictions.npz')};e.write(path/'report.json',rec);return base+np.mean(cor,0),base,rec

def run(split):
 check();d=load(split);path=_repo_path(OUT, split);path.mkdir(exist_ok=False);start=time.perf_counter();pred={arm:np.full(400,np.nan) for arm in ARMS};frozen={arm:np.full(400,np.nan) for arm in ARMS};bases={arm:np.full(400,np.nan) for arm in ARMS};records={};calibrators=[]
 for k in range(5):
  T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);mu=d['y'][T].mean();_,oldcal=affine(mixture(d['middleP'][k],'full'),d['y'][T],mixture(d['P'][V],'full'));calibrators.append(oldcal)
  for arm in ARMS:
   dest=path/f'{arm}-fold-{k}';pred[arm][V],bases[arm][V],rr=fold(d,k,arm,dest);records[dest.name]=e.sha(dest/'report.json');drop=~MASKS[arm];amount=.9*oldcal['slope']*((d['P'][V][:,drop]-mu)@W[drop]);amount+=.1*(d['short'][V]-mu) if arm=='no_DINO' else 0.;frozen[arm][V]=d['reference013'][V]-amount
  e.emit({'split':split,'outer_complete':k,'elapsed_seconds':time.perf_counter()-start})
 pred['no_new_residual']=d['reference011'];frozen['no_new_residual']=d['reference011'];pred['reference013']=d['reference013']
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],components=d['P'],short=d['short'],original_correction=d['correction'],**pred,**{f'frozen_{a}':p for a,p in frozen.items()},**{f'base_{a}':p for a,p in bases.items()})
 rec={'split':split,'scores':{name:e.rmse(d['y'],p) for name,p in pred.items()},'frozen_diagnostic_scores':{name:e.rmse(d['y'],p) for name,p in frozen.items()},'original_calibrators':calibrators,'fold_reports_sha256':records,'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(path/'report.json',rec);check();e.emit(rec)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare',*e.SPLITS]);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
