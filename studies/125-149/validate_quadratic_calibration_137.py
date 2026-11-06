"""Independent augmented least-squares and kernel verification; no holdout/test."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,argparse,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import quadratic_calibration_137 as q
import validate_nonlinear_residual_115 as vn
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
e=q.e;a=q.a;n=q.x.s.old.n if hasattr(q.x.s.old,'n') else a.n

def independent(p,y,v):
 mu=np.mean(p);sd=np.std(p);z=(p-mu)/sd;zv=(v-mu)/sd;A=np.column_stack([np.ones(len(p)),z]);orth=np.linalg.lstsq(A,z*z,rcond=None)[0];b=z*z-A@orth;bs=np.sqrt(np.mean(b*b));deg=bs<1e-12;Q=np.zeros(len(p)) if deg else b/bs;VQ=np.zeros(len(v)) if deg else (zv*zv-np.column_stack([np.ones(len(v)),zv])@orth)/bs;X=np.column_stack([A,Q]);aug=np.vstack([X,[0,0,np.sqrt(len(p))]]);coef=np.linalg.lstsq(aug,np.r_[y,0.],rcond=None)[0];out=np.column_stack([np.ones(len(v)),zv,VQ])@coef;return out

def validate(split):
 start=time.perf_counter();q.check();d=q.data(split);root=q.OUT/split;rec=e.read(root/'report.json');saved=a.readnp(root/'predictions.npz',rec['predictions_sha256']);np.testing.assert_array_equal(saved['ids'],d['ids']);np.testing.assert_array_equal(saved['y'],d['y']);models=0;maxerr=0.;maxcal=0.;extra=[]
 for k in range(5):
  dd,_=q.x.replace(d,split,k,q.x.OUT/split/f'cnn-fold-{k}',False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);path=root/f'fold-{k}';fr=e.read(path/'report.json');assert e.sha(path/'report.json')==rec['fold_report_sha256'][f'fold-{k}/report.json'];z=a.readnp(path/'predictions.npz',fr['predictions_sha256']);raw=a.mixture(dd['P'][V],q.x.ARM);train=a.mixture(dd['middleP'][k],q.x.ARM);actual=q.predict(fr['outer_model'],raw);manual=independent(train,d['y'][T],raw);err=float(abs(actual-manual).max());assert err<1e-9;maxcal=max(maxcal,err);base=.9*actual+.1*dd['short'][V];np.testing.assert_array_equal(base,z['base']);corr=[];extra.append({'fold':k,'outer_prediction_outside_meta_train_range':int(((raw<train.min())|(raw>train.max())).sum()),'curvature':fr['outer_model']['curvature'],'extra_df':fr['outer_model']['extra_effective_df']})
  for j in range(3):
   A,B,P,short,_=dd['cells'][k,j];pp=a.mixture(P,q.x.ARM);tt=a.mixture(dd['deepP'][k,j],q.x.ARM);mm=fr['anchor_models'][j];calc=q.predict(mm,pp);manual=independent(tt,d['y'][A],pp);err=float(abs(calc-manual).max());assert err<1e-9;maxcal=max(maxcal,err);anchor=.9*calc+.1*short;bp=path/f'block-{j}';br=e.read(bp/'report.json');assert e.sha(bp/'report.json')==fr['block_report_sha256'][f'block-{j}/report.json'];bz=a.readnp(bp/'predictions.npz',br['predictions_sha256']);np.testing.assert_array_equal(anchor,bz['anchor_predictions']);res=d['y'][B]-anchor;np.testing.assert_array_equal(res,bz['residual']);ip=np.empty_like(bz['inner_predictions']);folds=list(GroupKFold(3,shuffle=True,random_state=a.r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]))
   for t,(it,iv) in enumerate(folds):
    assert br['splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
    for ci,kind in enumerate(a.n.KINDS):
     f=bp/f'inner-{t}-{kind}.joblib';assert e.sha(f)==br['models_sha256'][f.name];m=joblib.load(_repo_io(f));np.testing.assert_array_equal(m['train_indices'],B[it]);np.testing.assert_array_equal(m['validation_indices'],B[iv]);actual=a.n.predict(m['model'],d['X'][B[iv]]);manual=vn.manual(m['model'],d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);err=float(abs(actual-manual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,bz['inner_predictions'][ci,iv]);ip[ci,iv]=manual;models+=1
   scores=np.mean((ip-res)**2,axis=1);np.testing.assert_allclose(scores,list(br['inner_MSE'].values()),rtol=0,atol=2e-9);chosen=a.n.KINDS[int(scores.argmin())];assert chosen==br['chosen'];f=bp/'selected.joblib';assert e.sha(f)==br['models_sha256'][f.name];m=joblib.load(_repo_io(f));np.testing.assert_array_equal(m['train_indices'],B);manual=vn.manual(m['model'],d['X'][B],res,d['X'][V],d['blocks']);actual=a.n.predict(m['model'],d['X'][V]);err=float(abs(actual-manual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,bz['correction']);corr.append(actual);models+=1
  np.testing.assert_array_equal(base+np.mean(corr,axis=0),saved['primary'][V])
 # Current new stages are invariant to outer-V targets; cached base exclusion was independently certified in133.
 dd,_=q.x.replace(d,split,0,q.x.OUT/split/'cnn-fold-0',False);V=np.flatnonzero(d['fid']==0);pert=dict(dd);pert['y']=dd['y'].copy();pert['y'][V]+=10000;pred,pr=q.fold(pert,0,q.OUT/f'outer-poison-{split}');np.testing.assert_array_equal(pred,saved['primary'][V]);original=e.read(root/'fold-0/report.json');assert pr['outer_model']==original['outer_model'] and pr['anchor_models']==original['anchor_models'] and pr['selected']==original['selected']
 A,B,_,_,_=dd['cells'][0,0];pert['y']=dd['y'].copy();pert['y'][B]+=50000;tt=a.mixture(dd['deepP'][0,0],q.x.ARM);assert q.fit(tt,pert['y'][A])==original['anchor_models'][0]
 fixturep=np.linspace(-3,3,50);fixturey=2+fixturep+.4*fixturep**2;fv=np.array([-4.,0.,4.]);fm=q.fit(fixturep,fixturey);np.testing.assert_allclose(q.predict(fm,fv),independent(fixturep,fixturey,fv),atol=1e-12,rtol=0);assert abs(fm['extra_effective_df']-.5)<1e-12
 q.check();out={'passed':True,'split':split,'independent_augmented_LS_max_error':maxcal,'independent_residual_kernel_max_error':maxerr,'residual_models_verified':models,'outer_target_poison_exact':True,'innerB_anchor_target_poison_exact':True,'base_cache_poison_certificate':e.sha(q.x.OUT/f'validation-{split}.json'),'new_neural_fits':0,'extrapolation_diagnostics_not_selection':extra,'report_sha256':e.sha(root/'report.json'),'source_sha256':e.sha(Path(__file__)),'seconds':time.perf_counter()-start};e.write(q.OUT/f'validation-{split}.json',out);e.emit(out)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('split',choices=q.e.SPLITS);validate(p.parse_args().split)
