"""Independent constrained optimization, OLS, residual solves, and label interventions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import constrained_weights_125 as q
import numpy as np,joblib
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
a=q.a;e=q.e;n=q.n;r=q.r;vn=q.protected.vn

def manual_meta(m,P,y,Q):
 wm=m['weight_model'];mean=P[:,1:].mean(0);ys=float(y.std());ym=float(y.mean());np.testing.assert_array_equal(mean,wm['prediction_mean']);assert ys==wm['target_std'] and ym==wm['target_mean'] and len(y)==wm['n_fit'];Z=(P[:,1:]-mean)/ys;t=(y-ym)/ys
 def obj(w):return float(np.mean((Z@w-t)**2)+np.sum((w-q.PRIOR)**2))
 def jac(w):return 2*Z.T@(Z@w-t)/len(y)+2*(w-q.PRIOR)
 sol=minimize(obj,q.PRIOR.copy(),jac=jac,method='SLSQP',bounds=[(0,1)]*4,constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(4)},options={'ftol':1e-13,'maxiter':1000});assert sol.success,sol.message;w=np.array(wm['weights']);assert w.min()>=0 and abs(w.sum()-1)<1e-12;err=float(abs(sol.x-w).max());assert err<1e-6;assert abs(obj(w)-wm['objective'])<1e-12 and wm['objective']<=wm['prior_objective']+1e-12
 ip=P[:,1:]@w;coef=np.linalg.lstsq(np.column_stack([np.ones(len(y)),ip-ip.mean()]),y,rcond=None)[0];pred=coef[0]+coef[1]*(Q[:,1:]@w-ip.mean());cal=m['calibrator'];assert abs(coef[0]-cal['target_center'])<1e-10 and abs(coef[1]-cal['slope'])<1e-10;return pred,err

def fixtures():
 q.check();rng=np.random.default_rng(20271101);errors=[]
 for k in range(10):
  P=rng.normal(size=(40,4))*np.array([1.,2.,3.,4.]);y=P[:,0]+rng.normal(size=40)*.4;full=np.column_stack([np.zeros(40),P]);m=q.fit_weights(P,y);_,cal=a.affine(P@np.array(m['weights']),y,P@np.array(m['weights']));_,err=manual_meta({'weight_model':m,'calibrator':cal},full,y,full);errors.append(err);shift=q.fit_weights(P+np.array([100,30,-20,7]),y+4);scale=q.fit_weights(P*7,y*7);np.testing.assert_allclose(m['weights'],shift['weights'],atol=1e-12,rtol=0);np.testing.assert_allclose(m['weights'],scale['weights'],atol=1e-12,rtol=0)
 P=np.tile(rng.normal(size=(30,1)),(1,4));m=q.fit_weights(P,rng.normal(size=30));np.testing.assert_allclose(m['weights'],q.PRIOR,atol=1e-12,rtol=0)
 d=q.load('original');path=q.OUT/'fixed014-control';p,_,_=a.fold(d,0,'no_baseline795',path);V=np.flatnonzero(d['fid']==0);err=float(abs(p-d['reference014'][V]).max());assert err<1e-10
 out={'passed':True,'SLSQP_fixture_max_weight_error':max(errors),'shift_and_scale_invariance':True,'identical_predictors_return_prior':True,'fixed014_full_fold_refit_max_error':err,'source_sha256':e.sha(Path(__file__))};e.write(q.OUT/'fixture-control-validation.json',out);e.emit(out)

def validate(split):
 q.check();d=q.load(split);out=q.OUT/split;report=e.read(out/'report.json');z=a.readnp(out/'predictions.npz',report['predictions_sha256']);assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['reference014'],d['reference014']);maxw=0.;maxkernel=0.;fixederr=0.;models=0;metas=0
 for k in range(5):
  T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);fd=out/f'fold-{k}';rec=e.read(fd/'report.json');assert e.sha(fd/'report.json')==report['fold_reports_sha256'][f'fold-{k}/report.json'];assert rec['outer_meta']['train_indices']==T.tolist() and rec['outer_meta']['validation_indices']==V.tolist();pp,err=manual_meta(rec['outer_meta'],d['middleP'][k],d['y'][T],d['P'][V]);maxw=max(maxw,err);metas+=1;base=.9*pp+.1*d['short'][V];cor=[];ff=a.readnp(fd/'predictions.npz',rec['predictions_sha256']);np.testing.assert_allclose(base,ff['base'],atol=1e-10,rtol=0)
  oldfd=q.protected.study.OUT/split/f'no_baseline795-fold-{k}';old=a.readnp(oldfd/'predictions.npz');fixed,_=a.affine(a.mixture(d['middleP'][k],'no_baseline795'),d['y'][T],a.mixture(d['P'][V],'no_baseline795'));err=float(abs(.9*fixed+.1*d['short'][V]-old['base']).max());assert err<1e-10;fixederr=max(fixederr,err)
  for j in range(3):
   A,B,P,short,_=d['cells'][k,j];m=rec['anchor_meta'][j];assert m['train_indices']==A.tolist() and m['validation_indices']==B.tolist();ap,err=manual_meta(m,d['deepP'][k,j],d['y'][A],P);maxw=max(maxw,err);metas+=1;anchor=.9*ap+.1*short;bp=fd/f'block-{j}';br=e.read(bp/'report.json');assert e.sha(bp/'report.json')==rec['block_reports_sha256'][f'block-{j}/report.json'];zz=a.readnp(bp/'predictions.npz',br['predictions_sha256']);np.testing.assert_allclose(anchor,zz['anchor_predictions'],atol=1e-10,rtol=0);res=d['y'][B]-zz['anchor_predictions'];np.testing.assert_array_equal(res,zz['residual']);ip=np.empty_like(zz['inner_predictions']);folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]))
   fixed,_=a.affine(a.mixture(d['deepP'][k,j],'no_baseline795'),d['y'][A],a.mixture(P,'no_baseline795'));old=a.readnp(oldfd/f'block-{j}/predictions.npz');err=float(abs(.9*fixed+.1*short-old['anchor_predictions']).max());assert err<1e-10;fixederr=max(fixederr,err)
   for t,(it,iv) in enumerate(folds):
    assert br['splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]]) and not set(A)&set(B)
    for ci,kind in enumerate(n.KINDS):
     file=bp/f'inner-{t}-{kind}.joblib';assert e.sha(file)==br['models_sha256'][file.name];saved=joblib.load(_repo_io(file));np.testing.assert_array_equal(saved['train_indices'],B[it]);np.testing.assert_array_equal(saved['validation_indices'],B[iv]);manual=vn.manual(saved['model'],d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);actual=n.predict(saved['model'],d['X'][B[iv]]);np.testing.assert_array_equal(actual,zz['inner_predictions'][ci,iv]);err=float(abs(manual-actual).max());assert err<2e-9;maxkernel=max(maxkernel,err);ip[ci,iv]=manual;models+=1
   scores=np.mean((ip-res[None,:])**2,1);np.testing.assert_allclose(scores,list(br['inner_MSE'].values()),atol=2e-9,rtol=0);chosen=n.KINDS[int(scores.argmin())];assert chosen==br['chosen'];file=bp/'selected.joblib';assert e.sha(file)==br['models_sha256'][file.name];saved=joblib.load(_repo_io(file));np.testing.assert_array_equal(saved['train_indices'],B);assert saved['model']['kind']==chosen;manual=vn.manual(saved['model'],d['X'][B],res,d['X'][V],d['blocks']);actual=n.predict(saved['model'],d['X'][V]);np.testing.assert_array_equal(actual,zz['correction']);err=float(abs(manual-actual).max());assert err<2e-9;maxkernel=max(maxkernel,err);cor.append(actual);models+=1
  np.testing.assert_array_equal(np.stack(cor),ff['corrections']);np.testing.assert_allclose(base+np.mean(cor,0),z['primary'][V],atol=1e-10,rtol=0)
 out={'passed':True,'split':split,'independent_weight_affine_fits':metas,'SLSQP_max_weight_error':maxw,'residual_models_reloaded':models,'independent_kernel_max_error':maxkernel,'fixed014_base_anchor_reconstruction_error':fixederr,'report_sha256':e.sha(q.OUT/split/'report.json')};e.write(q.OUT/f'validation-{split}.json',out);q.check();e.emit(out)

def replay():
 q.check();d=q.load('original');alter=dict(d);alter['y']=d['y'].copy();V=np.flatnonzero(d['fid']==0);alter['y'][V]+=10000;pp,rr=q.fold(alter,0,q.OUT/'outer-label-replay');path=q.OUT/'original/fold-0';old=e.read(path/'report.json');assert rr['outer_meta']==old['outer_meta'] and rr['anchor_meta']==old['anchor_meta'] and rr['selected']==old['selected']
 with np.load(_repo_io(path/'predictions.npz')) as z:err=float(abs(pp-z['primary']).max());assert err<1e-10
 # A excludes allB, including each residual-inner validation fold; weightfit cannotsee B labels.
 A,B,P,short,_=d['cells'][0,0];alt=d['y'].copy();alt[B]+=54321;_,meta=q.meta(d['deepP'][0,0],alt[A],P);assert meta['weight_model']==old['anchor_meta'][0]['weight_model'] and meta['calibrator']==old['anchor_meta'][0]['calibrator']
 bp=path/'block-0';br=e.read(bp/'report.json');z=a.readnp(bp/'predictions.npz');res=z['residual'].copy();it=np.array(br['splits'][0]['train_indices']);iv=np.array(br['splits'][0]['validation_indices']);index={int(i):j for j,i in enumerate(B)};res[[index[int(i)] for i in iv]]+=10000;maxerr=0.
 for kind in n.KINDS:
  m=n.fit(d['X'][it],res[[index[int(i)] for i in it]],d['blocks'],kind);oldm=joblib.load(_repo_io(bp/f'inner-0-{kind}.joblib'))['model'];diff=float(abs(n.predict(m,d['X'][V])-n.predict(oldm,d['X'][V])).max());assert diff<1e-10;maxerr=max(maxerr,diff)
 result={'passed':True,'outer80_meta_models_and_choices_exact':True,'outer80_prediction_max_error':err,'anchor_meta_excludes_all_residual_block_labels':True,'residual_inner_validation_5fixedfit_max_error':maxerr,'upstream112_training_certificate_reused':e.sha(e.OUT/'validation-replay.json'),'source_sha256':e.sha(Path(__file__))};e.write(q.OUT/'replay-validation.json',result);q.check();e.emit(result)
if __name__=='__main__':
 q.protected.torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['fixtures',*e.SPLITS,'replay']);x=p.parse_args();{'fixtures':fixtures,'replay':replay}.get(x.mode,lambda:validate(x.mode))()
