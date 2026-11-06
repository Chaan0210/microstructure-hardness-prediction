"""Independent balanced-kernel algebra, nested selection and feature audit for101."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,time,argparse,types
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,joblib
from scipy.spatial.distance import pdist,cdist,squareform
from scipy.linalg import solve
from sklearn.model_selection import GroupKFold
from skimage.measure import regionprops
from PIL import Image
import frame_features_101 as s

def manual(X,y,Z,blocks,width,alpha):
 lo=X.min(0);span=np.ptp(X,axis=0);span=np.where(span<10*np.finfo(float).eps,1.,span)
 A=np.clip((X-lo)/span,0,1);B=np.clip((Z-lo)/span,0,1);mu=A.mean(0);std=A.std(0);std=np.where(std<10*np.finfo(float).eps,1.,std)
 w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);A=(A-mu)/std*w;B=(B-mu)/std*w
 dist=pdist(A,'sqeuclidean');gamma=1/max(float(np.median(dist[dist>1e-12]))*width**2,1e-12);K=np.exp(-gamma*squareform(dist));coef=solve(K+alpha*np.eye(len(y)),(y-y.mean())/max(y.std(),1e-8),assume_a='pos')
 return np.exp(-gamma*cdist(B,A,'sqeuclidean'))@coef*max(y.std(),1e-8)+y.mean()

def main(split):
 start=time.perf_counter();p=s.check();ids,y,fid,ref,g=s.inputs(split);r=s.read(s.OUT/split/'report.json');ex=s.read(s.OUT/'extraction.json');assert s.sha(s.OUT/'development_features.npz')==ex['features_sha256']
 with np.load(_repo_io(s.OUT/'development_features.npz'),allow_pickle=False) as z:X=z['X'].copy();raw=z['raw_four_config_diameters'].copy();assert z['ids'].tolist()==ids
 path=s.OUT/split/'development_predictions.npz';assert s.sha(path)==r['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 np.testing.assert_array_equal(arr['ids'],ids);np.testing.assert_array_equal(arr['y'],y);np.testing.assert_array_equal(arr['fold'],fid);np.testing.assert_array_equal(arr['reference011'],ref)
 np.testing.assert_array_equal(X[:,-2:],np.column_stack([np.median(raw,axis=1),np.ptp(raw,axis=1)]))
 with np.load(_repo_path(ROOT, 'artifacts/run-022/development_features.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(X[:,:57],z['X'])
 maxerr=0.;innererr=0.;reloaderr=0.;replay={}
 for arm,dim in s.DIMS.items():
  for k in range(5):
   path=s.OUT/split/f'{arm}-fold{k}.joblib';assert s.sha(path)==r['models_sha256'][path.name];saved=joblib.load(_repo_io(path));m=saved['model'];tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);A=X[tr,:dim];Z=X[va,:dim];blocks=p['blocks'][:dim]
   np.testing.assert_array_equal(saved['train_indices'],tr);np.testing.assert_array_equal(saved['validation_indices'],va);assert saved['train_ids']==[ids[i] for i in tr] and saved['validation_ids']==[ids[i] for i in va]
   assert not set(g[tr])&set(g[va]);np.testing.assert_array_equal(m.bound_.data_min_,A.min(0));np.testing.assert_array_equal(m.bound_.data_max_,A.max(0));np.testing.assert_allclose(m.scale_.mean_,m.bound_.transform(A).mean(0),rtol=0,atol=1e-12)
   np.testing.assert_allclose(m.scale_.var_,m.bound_.transform(A).var(0),rtol=0,atol=1e-12);assert m.ym_==y[tr].mean() and m.ys_==max(y[tr].std(),1e-8)
   pred=m.predict(Z);reloaderr=max(reloaderr,float(abs(pred-arr[arm][va]).max()));np.testing.assert_array_equal(pred,arr[arm][va]);ind=manual(A,y[tr],Z,blocks,m.bandwidth,m.alpha);maxerr=max(maxerr,float(abs(ind-pred).max()));np.testing.assert_allclose(ind,pred,rtol=0,atol=1e-7)
   inner=list(GroupKFold(3,shuffle=True,random_state=20261005+k).split(A,groups=g[tr]));scores=[]
   for width in (.25,1.,4.):
    for alpha in (.01,.1,1.):
     ip=np.full(len(tr),np.nan)
     for it,iv in inner:
      assert not set(g[tr[it]])&set(g[tr[iv]]);ip[iv]=manual(A[it],y[tr[it]],A[iv],blocks,width,alpha)
     scores.append({'bandwidth':width,'alpha':alpha,'inner_rmse':s.rmse(y[tr],ip)})
   for a,b in zip(scores,saved['detail']['inner_scores']):innererr=max(innererr,abs(a['inner_rmse']-b['inner_rmse']));assert abs(a['inner_rmse']-b['inner_rmse'])<1e-7
   chosen=min(scores,key=lambda z:z['inner_rmse']);assert chosen['bandwidth']==m.bandwidth and chosen['alpha']==m.alpha
   if k==0 and split=='original':
    pert=y.copy();pert[va]+=12345.;again,mm,detail=s.fv.kernel_fold(X[:,:dim],pert,g,blocks,tr,va,k);np.testing.assert_array_equal(again,pred);np.testing.assert_array_equal(mm.coef_,m.coef_);assert detail==saved['detail'];replay[arm]=True
  np.testing.assert_array_equal(arr[arm+'_blend'],.9*ref+.1*arr[arm])
 for key in r['scores']:assert abs(s.rmse(y,arr[key])-r['scores'][key]['rmse'])<1e-12
 assert s.gate(y,arr,fid,ids)==r['gate']
 feature_err=None
 if split=='original':
  feature_err=0.
  for i in (0,199,399):
   with Image.open(_repo_path(ROOT, 'data/train', f'{ids[i]}.png')) as im:a=np.asarray(im,dtype=np.uint8).copy()
   values=[]
   def observe(seg,x):
    rr=[z for z in regionprops(seg) if z.area>=16];areas=np.array([z.area for z in rr]);diam=np.array([z.equivalent_diameter_area for z in rr]);values.append(float(np.average(diam,weights=areas)) if len(rr) else 0.);return s.dg._summarize(seg,x)
   gl=dict(s.dg.extract_denoised.__globals__);gl['_summarize']=observe;fn=types.FunctionType(s.dg.extract_denoised.__code__,gl);old,_,_=fn(a)
   np.testing.assert_allclose(old,X[i,-33:-2],rtol=0,atol=1e-12);np.testing.assert_allclose(values,raw[i],rtol=0,atol=1e-12);feature_err=max(feature_err,float(abs(np.array(values)-raw[i]).max()))
  for ident,digest in ex['image_sha256'].items():assert s.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==digest
 s.check();out={'passed':True,'split':split,'models_verified':10,'reload_error':reloaderr,'independent_kernel_prediction_max_error':maxerr,'independent_inner_rmse_max_error':innererr,'inner_selection_recomputed':True,'foldtrain_scalers_targets_ids_verified':True,'outer_validation_label_perturbation_replay':replay,'independent_regionprops_feature_max_error':feature_err,'old57_unchanged':True,'all400_old31_verified_at_extraction':True,'gate_recomputed':r['gate'],'report_sha256':s.sha(s.OUT/split/'report.json'),'seconds':time.perf_counter()-start,'protected011_verified':20,'holdout_test_access':False,'cost':0,'source_sha256':s.sha(Path(__file__))};s.write(_repo_path(ROOT, f'reports/frame-features-101-{split}-validation.json'),out);s.emit(out)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=s.SPLITS);main(ap.parse_args().split)
