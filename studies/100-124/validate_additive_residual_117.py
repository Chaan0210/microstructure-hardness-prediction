"""Independent per-block kernel construction/eigensolve and all selector scopes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,argparse,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import additive_residual_117 as a
import validate_nonlinear_residual_115 as v115
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
e=a.e;r=a.r

def manual(m,X,y,V,blocks):
 if m['kind'] not in a.ADD:return v115.manual(m,X,y,V,blocks)
 low=X.min(0);span=X.max(0)-low;span[span<10*np.finfo(float).eps]=1.;B=np.clip((X-low)/span,0,1);BV=np.clip((V-low)/span,0,1);mu=B.mean(0);var=B.var(0);sd=np.sqrt(var);sd[sd==0]=1.;w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);Z=(B-mu)/sd*w;ZV=(BV-mu)/sd*w
 np.testing.assert_array_equal(m['bound'].data_min_,low);np.testing.assert_allclose(m['scale'].mean_,mu,atol=1e-12,rtol=0);np.testing.assert_allclose(m['scale'].var_,var,atol=1e-12,rtol=0);np.testing.assert_allclose(m['Z'],Z,atol=1e-12,rtol=0);assert m['n_train']==len(y)
 C=np.zeros((len(X),len(X)));CV=np.zeros((len(V),len(X)));H=np.eye(len(X))-np.ones((len(X),len(X)))/len(X);names=list(dict.fromkeys(blocks));assert [p['name'] for p in m['parts']]==names
 for name,part in zip(names,m['parts']):
  cols=np.flatnonzero(np.array(blocks)==name);np.testing.assert_array_equal(cols,part['columns']);D=np.sum((Z[:,None,cols]-Z[None,:,cols])**2,2);tri=D[np.triu_indices(len(X),1)];pos=tri[tri>1e-12];gamma=1/float(np.median(pos)) if len(pos) else 1.;assert np.isclose(gamma,part['gamma'],rtol=1e-11,atol=1e-12);K=np.exp(-gamma*D);center=H@K@H;scale=float(np.trace(center)/len(X));scale=1. if scale<=1e-12 else scale;assert abs(scale-part['kernel_scale'])<1e-12;np.testing.assert_allclose(K.mean(0),part['column'],atol=1e-12,rtol=0);assert abs(K.mean()-part['grand'])<1e-12;VK=np.exp(-gamma*np.sum((ZV[:,None,cols]-Z[None,:,cols])**2,2));C+=center/scale;CV+=(VK-VK.mean(1)[:,None]-K.mean(0)[None,:]+K.mean())/scale
 C/=len(names);CV/=len(names);alpha=a.LAM[m['kind']]*len(y);assert m['alpha']==alpha;rs=float(np.sqrt(np.mean(y*y)));rs=rs if rs>1e-12 else 1.;assert rs==m['residual_scale'];ev,U=np.linalg.eigh(C);coef=U@((U.T@(y/rs))/(ev+alpha));np.testing.assert_allclose(coef,m['coef'],atol=2e-11,rtol=1e-10);df=float(np.sum(ev/(ev+alpha)));assert abs(df-m['effective_df'])<1e-10 and df<=1/a.LAM[m['kind']]+1e-8;return CV@coef*rs

def validate(split):
 a.check();d=r.load(split);out=a.OUT/split;report=e.read(out/'report.json');assert e.sha(out/'predictions.npz')==report['prediction_sha256'];start=time.perf_counter();count=0;maxerr=0.;assembled={f'{size}_{family}':np.full((3,400),np.nan) for size in ['full','half'] for family in a.SETS};dfs=[]
 with np.load(_repo_io(out/'predictions.npz'),allow_pickle=False) as z:ps={k:z[k].copy() for k in z.files}
 assert ps['ids'].tolist()==d['ids'];np.testing.assert_array_equal(ps['y'],d['y']);np.testing.assert_array_equal(ps['reference011'],d['reference011'])
 for k in range(5):
  va=np.flatnonzero(d['fid']==k)
  for j in range(3):
   A,whole,pa,trace=r.anchor(d,split,k,j)
   for size in ['full','half']:
    path=out/f'{size}-fold-{k}-block-{j}';rec=e.read(path/'report.json');assert e.sha(path/'report.json')==report['report_sha256'][f'{size}-fold-{k}-block-{j}/report.json'];assert e.sha(path/'predictions.npz')==rec['prediction_sha256'];assert trace==rec['anchor_trace'];B=whole if size=='full' else a.subset(d,whole,split,k,j);paB=pa[np.isin(whole,B)];res=d['y'][B]-paB
    with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:saved={key:z[key].copy() for key in z.files}
    np.testing.assert_array_equal(saved['block_indices'],B);np.testing.assert_array_equal(saved['whole_block_indices'],whole);np.testing.assert_array_equal(saved['anchor_indices'],A);np.testing.assert_array_equal(saved['residual'],res);assert not set(A)&set(B);ip=np.full((7,len(B)),np.nan);folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]))
    for t,(it,iv) in enumerate(folds):
     assert rec['inner_splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
     for ci,kind in enumerate(a.KINDS):
      fr=rec['files'][f'inner-{t}-{kind}'];file=_repo_path(ROOT, fr['source']);assert e.sha(file)==fr['sha256'];s=joblib.load(_repo_io(file));np.testing.assert_array_equal(s['train_indices'],B[it]);np.testing.assert_array_equal(s['validation_indices'],B[iv]);m=s['model'];assert m['kind']==kind;pp=manual(m,d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);re=a.predict(m,d['X'][B[iv]]);np.testing.assert_array_equal(re,saved['inner_predictions'][ci,iv]);err=float(abs(pp-re).max());assert err<2e-9;maxerr=max(maxerr,err);ip[ci,iv]=pp;count+=1
      if kind in a.ADD:dfs.append(m['effective_df'])
    scores=np.mean((ip-res[None,:])**2,1);np.testing.assert_allclose(scores,list(rec['inner_MSE'].values()),atol=2e-9,rtol=0)
    for family,kinds in a.SETS.items():
     choices=[a.KINDS.index(kind) for kind in kinds];chosen=a.KINDS[choices[int(np.argmin(scores[choices]))]];assert chosen==rec['selected'][family];fr=rec['files'][f'selected-{family}'];file=_repo_path(ROOT, fr['source']);assert e.sha(file)==fr['sha256'];s=joblib.load(_repo_io(file));np.testing.assert_array_equal(s['train_indices'],B);assert s['chosen']==chosen;m=s['model'];pp=manual(m,d['X'][B],res,d['X'][va],d['blocks']);np.testing.assert_allclose(pp,saved[family],atol=2e-9,rtol=0);np.testing.assert_array_equal(a.predict(m,d['X'][va]),saved[family]);assembled[f'{size}_{family}'][j,va]=saved[family];count+=1
 for key,c in assembled.items():np.testing.assert_array_equal(d['reference011']+c.mean(0),ps[key])
 with np.load(_repo_io(a.n.OUT/split/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(ps['full_linear'],z['reference113']);np.testing.assert_array_equal(ps['full_joint'],z['primary'])
 for key in report['scores']:assert float(np.mean((d['y']-ps[key])**2))==report['scores'][key]['MSE']
 a.check();v={'passed':True,'split':split,'models_reloaded':count,'independent_selected_choices_recomputed':90,'manual_transform_eigensolve_max_error':maxerr,'additive_effective_df_range':[min(dfs),max(dfs)],'full113_115_controls_exact':True,'all_group_and_anchor_exclusions_checked':True,'half_subset_fixed_without_labels':True,'report_sha256':e.sha(out/'report.json'),'seconds':time.perf_counter()-start};e.write(a.OUT/f'validation-{split}.json',v);e.emit(v)

def leakage():
 a.check();d=r.load('original');pert=dict(d);pert['y']=d['y'].copy();ix=np.flatnonzero(d['fid']==0);pert['y'][ix]=12345+np.arange(len(ix));out=a.OUT/'leakage-replay';out.mkdir(exist_ok=False);checks=0
 for j in range(3):
  for size in ['full','half']:
   pp,new=a.block(pert,'original',0,j,size,out/f'{size}-{j}');old=a.OUT/'original'/f'{size}-fold-0-block-{j}';rec=e.read(old/'report.json');assert new['selected']==rec['selected'] and new['inner_MSE']==rec['inner_MSE']
   with np.load(_repo_io(old/'predictions.npz'),allow_pickle=False) as z:
    for family in a.SETS:np.testing.assert_array_equal(pp[family],z[family])
    B=z['block_indices'];res=z['residual'].copy()
   it,iv=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+j).split(d['X'][B],res,d['g'][B]))[0];res[iv]+=54321
   for kind in a.ADD:
    m=a.fit(d['X'][B[it]],res[it],d['blocks'],kind);fr=rec['files'][f'inner-0-{kind}'];saved=joblib.load(_repo_path(ROOT, fr['source']))['model'];np.testing.assert_array_equal(m['coef'],saved['coef']);assert m['residual_scale']==saved['residual_scale'];checks+=1
 v={'passed':True,'outer80_perturbed_full_half_all3blocks_exact':True,'inner_validation_labels_perturbed_fits_exact':checks,'upstream_exclusion_certificate_sha256':e.sha(r.prior.OUT/'validation-replay.json')};e.write(a.OUT/'leakage-validation.json',v);e.emit(v)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=[*e.SPLITS,'leakage']);x=p.parse_args();leakage() if x.mode=='leakage' else validate(x.mode)
