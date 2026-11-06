"""Raw-statistic transforms and eigensystem kernel solve independent of training solver."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,argparse,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nonlinear_residual_115 as n
import validate_residual_physics_113 as vr
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
e=n.e;r=n.r

def manual(m,X,y,V,blocks):
 if m['kind'] in r.CANDIDATES:return vr.manual(m,X,y,V,blocks)
 low=X.min(0);span=X.max(0)-low;span[span<10*np.finfo(float).eps]=1.;B=np.clip((X-low)/span,0,1);BV=np.clip((V-low)/span,0,1);mu=B.mean(0);var=B.var(0);sd=np.sqrt(var);sd[sd==0]=1.;w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);Z=(B-mu)/sd*w;ZV=(BV-mu)/sd*w
 np.testing.assert_array_equal(m['bound'].data_min_,low);np.testing.assert_allclose(m['scale'].mean_,mu,atol=1e-12,rtol=0);np.testing.assert_allclose(m['scale'].var_,var,atol=1e-12,rtol=0);np.testing.assert_allclose(m['Z'],Z,atol=1e-12,rtol=0);np.testing.assert_array_equal(m['weights'],w)
 D=np.sum((Z[:,None,:]-Z[None,:,:])**2,2);tri=D[np.triu_indices(len(X),1)];pos=tri[tri>1e-12];gamma=1/float(np.median(pos)) if len(pos) else 1.;assert abs(gamma-m['gamma'])<1e-12;K=np.exp(-gamma*D);H=np.eye(len(X))-np.ones((len(X),len(X)))/len(X);C=H@K@H;scale=float(np.trace(C)/len(X));scale=1. if scale<=1e-12 else scale;assert abs(scale-m['kernel_scale'])<1e-12;C/=scale;np.testing.assert_allclose(m['column'],K.mean(0),atol=1e-12,rtol=0);assert abs(m['grand']-K.mean())<1e-12
 rs=float(np.sqrt(np.mean(y*y)));rs=rs if rs>1e-12 else 1.;assert rs==m['residual_scale'] and m['alpha']==n.LAM[m['kind']]*len(y);values,vectors=np.linalg.eigh(C);dual=vectors@((vectors.T@(y/rs))/(values+m['alpha']));np.testing.assert_allclose(dual,m['coef'],atol=2e-11,rtol=1e-10);df=float(np.sum(values/(values+m['alpha'])));assert abs(df-m['effective_df'])<1e-10 and df<=1/n.LAM[m['kind']]+1e-8
 VK=np.exp(-gamma*np.sum((ZV[:,None,:]-Z[None,:,:])**2,2));cross=(VK-VK.mean(1)[:,None]-K.mean(0)[None,:]+K.mean())/scale;return cross@dual*rs

def validate(split):
 n.check();d=r.load(split);out=n.OUT/split;report=e.read(out/'report.json');assert e.sha(out/'predictions.npz')==report['predictions_sha256'];start=time.perf_counter();maxerr=0.;count=0;cor=np.full((3,400),np.nan);dfs=[]
 with np.load(_repo_io(out/'predictions.npz'),allow_pickle=False) as z:ps={k:z[k].copy() for k in z.files}
 assert ps['ids'].tolist()==d['ids'];np.testing.assert_array_equal(ps['y'],d['y']);np.testing.assert_array_equal(ps['reference011'],d['reference011'])
 with np.load(_repo_io(r.OUT/split/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(ps['reference113'],z['primary'])
 for k in range(5):
  va=np.flatnonzero(d['fid']==k)
  for j in range(3):
   path=out/f'fold-{k}-block-{j}';br=e.read(path/'report.json');assert e.sha(path/'report.json')==report['block_reports_sha256'][f'fold-{k}-block-{j}/report.json'];assert e.sha(path/'predictions.npz')==br['predictions_sha256'];A,B,pa,trace=r.anchor(d,split,k,j);assert trace==br['anchor_trace'];res=d['y'][B]-pa
   with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:saved={a:z[a].copy() for a in z.files}
   np.testing.assert_array_equal(saved['anchor_indices'],A);np.testing.assert_array_equal(saved['block_indices'],B);np.testing.assert_array_equal(saved['residual'],res);ip=np.empty_like(saved['inner_predictions']);folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]))
   for t,(it,iv) in enumerate(folds):
    assert br['inner_splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]]) and not set(A)&set(B)
    for ci,kind in enumerate(n.KINDS):
     file=(r.OUT/split/f'fold-{k}/block-{j}' if kind in r.CANDIDATES else path)/f'inner-{t}-{kind}.joblib';mrec=joblib.load(_repo_io(file));np.testing.assert_array_equal(mrec['train_indices'],B[it]);np.testing.assert_array_equal(mrec['validation_indices'],B[iv]);m=mrec['model'];assert m['kind']==kind
     if kind not in r.CANDIDATES:assert e.sha(file)==br['models_sha256'][file.name];dfs.append(m['effective_df'])
     pp=manual(m,d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);re=n.predict(m,d['X'][B[iv]]);np.testing.assert_array_equal(re,saved['inner_predictions'][ci,iv]);err=float(abs(pp-re).max());assert err<2e-9;maxerr=max(maxerr,err);ip[ci,iv]=pp;count+=1
   scores=[float(np.mean((res-p)**2)) for p in ip];np.testing.assert_allclose(scores,list(br['inner_MSE'].values()),atol=2e-9,rtol=0);chosen=n.KINDS[int(np.argmin(scores))];assert chosen==br['chosen'];assert e.sha(path/'selected.joblib')==br['selected_sha256'];m=joblib.load(_repo_io(path/'selected.joblib'));np.testing.assert_array_equal(m['train_indices'],B);assert m['chosen']==chosen;pp=manual(m['model'],d['X'][B],res,d['X'][va],d['blocks']);np.testing.assert_allclose(pp,saved['correction'],atol=2e-9,rtol=0);np.testing.assert_array_equal(n.predict(m['model'],d['X'][va]),saved['correction']);cor[j,va]=saved['correction'];count+=1
 np.testing.assert_array_equal(cor,ps['corrections']);np.testing.assert_array_equal(d['reference011']+cor.mean(0),ps['primary'])
 for name in ['reference011','reference113','primary']:assert float(np.mean((d['y']-ps[name])**2))==report['scores'][name]['MSE']
 n.check();v={'passed':True,'split':split,'models_reloaded':count,'inner_choices_recomputed':15,'independent_transform_eigensolve_max_error':maxerr,'nonlinear_inner_effective_df_range':[min(dfs),max(dfs)],'exact113_control':True,'report_sha256':e.sha(out/'report.json'),'seconds':time.perf_counter()-start};e.write(n.OUT/f'validation-{split}.json',v);e.emit(v)

def leakage():
 n.check();d=r.load('original');pert=dict(d);pert['y']=d['y'].copy();ix=np.flatnonzero(d['fid']==0);pert['y'][ix]=10000+np.arange(len(ix));out=n.OUT/'leakage-replay';out.mkdir(exist_ok=False);checks=0
 for j in range(3):
  pp,cc,new=n.block(pert,'original',0,j,out/f'block-{j}');old=n.OUT/'original'/f'fold-0-block-{j}';br=e.read(old/'report.json');assert new['chosen']==br['chosen'] and new['inner_MSE']==br['inner_MSE']
  with np.load(_repo_io(old/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(pp,z['correction']);B=z['block_indices'];res=z['residual'].copy()
  folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+j).split(d['X'][B],res,d['g'][B]));it,iv=folds[0];res[iv]+=54321
  for kind in ['rbf1','rbf01']:
   m=n.fit(d['X'][B[it]],res[it],d['blocks'],kind);saved=joblib.load(_repo_io(old/f'inner-0-{kind}.joblib'))['model'];np.testing.assert_array_equal(m['coef'],saved['coef']);assert m['gamma']==saved['gamma'] and m['residual_scale']==saved['residual_scale'];checks+=1
 v={'passed':True,'outer80_label_perturbation_exact_predictions_and_choices':True,'inner_validation_label_perturbation_exact_fixed_fits':checks,'upstream112_exclusion_certificate_reused':e.sha(r.prior.OUT/'validation-replay.json'),'new_upstream_training':False};e.write(n.OUT/'leakage-validation.json',v);e.emit(v)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=[*e.SPLITS,'leakage']);a=p.parse_args();leakage() if a.mode=='leakage' else validate(a.mode)
