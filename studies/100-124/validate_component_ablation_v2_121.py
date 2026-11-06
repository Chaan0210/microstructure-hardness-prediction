"""Independent OLS, manual kernel predictions, and label/path intervention audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,argparse,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import component_ablation_121 as a
import validate_nonlinear_residual_115 as vn
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
e=a.e;n=a.n;r=a.r

def mix(P,arm):
 keep=np.flatnonzero(a.MASKS[arm]);den=sum(float(a.W[i]) for i in keep);return sum(float(a.W[i])/den*P[:,i] for i in keep)
def ols(P,y,Q):
 design=np.column_stack([np.ones(len(P)),P-P.mean()]);coef=np.linalg.lstsq(design,y,rcond=None)[0];return coef[0]+coef[1]*(Q-P.mean()),coef

def validate(split):
 a.check();d=a.load(split);path=a.OUT/split;report=e.read(path/'report.json');assert e.sha(path/'predictions.npz')==report['predictions_sha256'];z=a.readnp(path/'predictions.npz');assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);count=0;maxerr=0.;choicecount=0;pred={arm:np.full(400,np.nan) for arm in a.ARMS}
 for k in range(5):
  T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k)
  for arm in a.ARMS:
   fd=path/f'{arm}-fold-{k}';rec=e.read(fd/'report.json');assert e.sha(fd/'report.json')==report['fold_reports_sha256'][fd.name];ff=a.readnp(fd/'predictions.npz',rec['predictions_sha256']);p009,coef=ols(mix(d['middleP'][k],arm),d['y'][T],mix(d['P'][V],arm));base=p009 if arm=='no_DINO' else .9*p009+.1*d['short'][V];np.testing.assert_allclose(base,ff['base'],atol=1e-10,rtol=0);cor=[]
   for j in range(3):
    A,B,P,short,_=d['cells'][k,j];ap,coef=ols(mix(d['deepP'][k,j],arm),d['y'][A],mix(P,arm));anchor=ap if arm=='no_DINO' else .9*ap+.1*short;bp=fd/f'block-{j}';br=e.read(bp/'report.json');assert e.sha(bp/'report.json')==rec['block_reports_sha256'][f'block-{j}/report.json'];zz=a.readnp(bp/'predictions.npz',br['predictions_sha256']);np.testing.assert_allclose(anchor,zz['anchor_predictions'],atol=1e-10,rtol=0);res=d['y'][B]-zz['anchor_predictions'];np.testing.assert_array_equal(res,zz['residual']);np.testing.assert_array_equal(zz['block_indices'],B);ip=np.empty_like(zz['inner_predictions']);folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]))
    for t,(it,iv) in enumerate(folds):
     assert br['splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
     for ci,kind in enumerate(n.KINDS):
      file=bp/f'inner-{t}-{kind}.joblib';assert e.sha(file)==br['models_sha256'][file.name];record=joblib.load(_repo_io(file));np.testing.assert_array_equal(record['train_indices'],B[it]);np.testing.assert_array_equal(record['validation_indices'],B[iv]);m=record['model'];assert m['kind']==kind;pp=vn.manual(m,d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);rp=n.predict(m,d['X'][B[iv]]);np.testing.assert_array_equal(rp,zz['inner_predictions'][ci,iv]);err=float(abs(pp-rp).max());assert err<2e-9;maxerr=max(maxerr,err);ip[ci,iv]=pp;count+=1
    scores=np.mean((ip-res[None,:])**2,1);np.testing.assert_allclose(scores,list(br['inner_MSE'].values()),atol=2e-9,rtol=0);chosen=n.KINDS[int(scores.argmin())];assert chosen==br['chosen'];choicecount+=1;file=bp/'selected.joblib';assert e.sha(file)==br['models_sha256'][file.name];record=joblib.load(_repo_io(file));np.testing.assert_array_equal(record['train_indices'],B);m=record['model'];assert m['kind']==chosen;pp=vn.manual(m,d['X'][B],res,d['X'][V],d['blocks']);rp=n.predict(m,d['X'][V]);np.testing.assert_array_equal(rp,zz['correction']);err=float(abs(pp-rp).max());assert err<2e-9;maxerr=max(maxerr,err);cor.append(rp);count+=1
   np.testing.assert_array_equal(np.stack(cor),ff['corrections']);np.testing.assert_allclose(base+np.mean(cor,0),ff['primary'],atol=1e-10,rtol=0);np.testing.assert_array_equal(ff['primary'],z[arm][V]);pred[arm][V]=ff['primary']
   mu=d['y'][T].mean();_,oc=ols(mix(d['middleP'][k],'full'),d['y'][T],mix(d['P'][V],'full'));removed=sum(a.W[i]*(d['P'][V,i]-mu) for i in np.flatnonzero(~a.MASKS[arm]));delta=.9*oc[1]*removed+(.1*(d['short'][V]-mu) if arm=='no_DINO' else 0);np.testing.assert_allclose(z[f'frozen_{arm}'][V],d['reference013'][V]-delta,atol=1e-10,rtol=0)
 np.testing.assert_array_equal(z['no_new_residual'],d['reference011']);np.testing.assert_array_equal(z['reference013'],d['reference013'])
 for key,score in report['scores'].items():assert e.rmse(d['y'],z[key])==score
 result={'passed':True,'models_reloaded':count,'choices_recomputed':choicecount,'independent_OLS_and_kernel_max_error':maxerr,'exact_no_residual_equals011':True,'frozen_intervention_independently_verified':True,'source_indices_and_removed_paths_audited':True,'report_sha256':e.sha(path/'report.json')};e.write(a.OUT/f'validation-{split}.json',result);a.check();e.emit(result)

def replay():
 a.check();d=a.load('original');path=a.OUT/'replays-v2';path.mkdir(exist_ok=False);V=np.flatnonzero(d['fid']==0);records={};A,B,_,_,old=d['cells'][0,0];cor,cr=a.fit_block(d,B,old,0,0,path/'incumbent-control');original=n.OUT/'original/fold-0-block-0';rec=e.read(original/'report.json');assert cr['chosen']==rec['chosen']
 with np.load(_repo_io(original/'predictions.npz')) as z:np.testing.assert_array_equal(z['inner_predictions'],np.load(_repo_io(path/'incumbent-control/predictions.npz'))['inner_predictions']);np.testing.assert_array_equal(z['correction'],cor)
 for arm in a.ARMS:
  pert=dict(d);pert['y']=d['y'].copy();pert['y'][V]+=10000;pp,_,rr=a.fold(pert,0,arm,path/(arm+'-outer-label'));oldpath=a.OUT/'original'/f'{arm}-fold-0';oldrec=e.read(oldpath/'report.json');assert rr['outer_calibrator']==oldrec['outer_calibrator'] and rr['anchor_calibrators']==oldrec['anchor_calibrators'] and rr['selected']==oldrec['selected']
  with np.load(_repo_io(oldpath/'predictions.npz')) as z:np.testing.assert_allclose(pp,z['primary'],atol=1e-10,rtol=0)
  # Replace removed branch outputs at every cached level; the reconstructed procedure must ignore them.
  alt=dict(d);alt['P']=d['P'].copy();alt['P'][:,~a.MASKS[arm]]=1e6;alt['middleP']={k:P.copy() for k,P in d['middleP'].items()};alt['deepP']={k:P.copy() for k,P in d['deepP'].items()};alt['cells']={}
  for P in [*alt['middleP'].values(),*alt['deepP'].values()]:P[:,~a.MASKS[arm]]=1e6
  for key,(AA,BB,P,short,old) in d['cells'].items():Q=P.copy();Q[:,~a.MASKS[arm]]=1e6;alt['cells'][key]=(AA,BB,Q,np.full_like(short,1e6) if arm=='no_DINO' else short,old)
  if arm=='no_DINO':alt['short']=np.full_like(d['short'],1e6)
  qq,_,ss=a.fold(alt,0,arm,path/(arm+'-removed-path'));np.testing.assert_allclose(pp,qq,atol=1e-10,rtol=0);assert rr['selected']==ss['selected']
  bp=oldpath/'block-0';br=e.read(bp/'report.json');z=a.readnp(bp/'predictions.npz');res=z['residual'].copy();B=z['block_indices'];it=np.array(br['splits'][0]['train_indices']);iv=np.array(br['splits'][0]['validation_indices']);lookup={int(i):j for j,i in enumerate(B)};res[[lookup[int(i)] for i in iv]]+=10000
  for kind in n.KINDS:
   m=n.fit(d['X'][it],res[[lookup[int(i)] for i in it]],d['blocks'],kind);saved=joblib.load(_repo_io(bp/f'inner-0-{kind}.joblib'))['model'];np.testing.assert_allclose(n.predict(m,d['X'][V]),n.predict(saved,d['X'][V]),atol=1e-10,rtol=0)
  records[arm]={'outer80_predictions_within_1e_10_calibrators_choices_exact':True,'removed_branch_output_intervention_all_levels_within_1e_10':True,'inner_label_intervention_fixed5fits_within_1e_10':True}
 result={'passed':True,'original_bitexact_replay_failure_preserved':True,'prediction_tolerance':1e-10,'reason':'Originalfirstouterlabelreplaydifferedby5.684341886080802e-14;usestheexistingindependentOLS1e-10tolerance. No changes toablationmodels orchoices.','source_sha256':e.sha(Path(__file__)),'incumbent115_control_inner5choices_predictions_exact':True,'arms':records,'upstream_independent_training_certificate_reused':e.sha(e.OUT/'validation-replay.json'),'scope':'Outer/deepbasepredictortrainingnotrerun;existing112certificatesandcachetrainmembershipverified. Newtrainablecalibratorsandresidualsreplayed.'};e.write(a.OUT/'replay-validation.json',result);a.check();e.emit(result)
if __name__=='__main__':
 a.torch.set_num_threads(2)
 p=argparse.ArgumentParser();p.add_argument('mode',choices=[*e.SPLITS,'replay']);x=p.parse_args();replay() if x.mode=='replay' else validate(x.mode)
