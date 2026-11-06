"""Independent OOF reconstruction and foldtrain audit for fixed096."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import sys,json,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import wavelet_scattering_096 as s
import numpy as np,joblib
from scipy.linalg import solve
from sklearn.model_selection import GroupKFold
from PIL import Image

def manual(X,y,Z,alpha):
 mu=X.mean(0);scale=X.std(0);scale=np.where(scale==0,1.,scale);A=(X-mu)/scale;B=(Z-mu)/scale
 aa=np.einsum('ij,ij->i',A,A);bb=np.einsum('ij,ij->i',B,B)
 D=np.maximum(aa[:,None]+aa[None,:]-2*A@A.T,0.);K=np.exp(-D/X.shape[1]);ym=y.mean();ys=y.std();coef=solve(K+alpha*np.eye(len(y)),(y-ym)/ys,assume_a='pos')
 Dz=np.maximum(bb[:,None]+aa[None,:]-2*B@A.T,0.);return np.exp(-Dz/X.shape[1])@coef*ys+ym

def run(split):
 start=time.perf_counter();s.check();ids,y,fid,ref,g=s.inputs(split);ex=s.read(s.OUT/'extraction.json')
 assert s.sha(s.OUT/'development_features.npz')==ex['features_sha256']
 with np.load(_repo_io(s.OUT/'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].copy()
 r=s.read(s.OUT/split/'report.json');path=s.OUT/split/'development_predictions.npz';assert s.sha(path)==r['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 assert arr['ids'].tolist()==ids;np.testing.assert_array_equal(arr['y'],y);np.testing.assert_array_equal(arr['fold'],fid);np.testing.assert_array_equal(arr['reference011'],ref)
 replay={};errors=[];models=0
 for arm,dim in s.DIMS.items():
  pred=np.full(400,np.nan)
  for k in range(5):
   p=s.OUT/split/f'{arm}-fold{k}.joblib';assert s.sha(p)==r['models_sha256'][p.name];m=joblib.load(_repo_io(p));tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k)
   np.testing.assert_array_equal(m['train_indices'],tr);np.testing.assert_array_equal(m['validation_indices'],va);assert m['train_ids']==[ids[i] for i in tr] and m['validation_ids']==[ids[i] for i in va]
   assert not set(g[tr])&set(g[va]);A=X[tr,:dim];Z=X[va,:dim];sc=m['model'].named_steps['scale'];kr=m['model'].named_steps['regressor']
   np.testing.assert_allclose(sc.mean_,A.mean(0),rtol=1e-12,atol=1e-14);np.testing.assert_allclose(sc.var_,A.var(0),rtol=1e-11,atol=1e-18);np.testing.assert_allclose(sc.scale_,np.where(A.std(0)==0,1.,A.std(0)),rtol=1e-11,atol=1e-14)
   assert m['target_mean']==float(y[tr].mean()) and m['target_std']==float(y[tr].std()) and kr.gamma==1./dim and kr.alpha==m['alpha']
   np.testing.assert_array_equal(kr.X_fit_,sc.transform(A))
   pred[va]=s.predict(m,Z);np.testing.assert_array_equal(pred[va],arr[arm][va])
   independent=manual(A,y[tr],Z,m['alpha']);err=float(abs(independent-pred[va]).max());errors.append(err);np.testing.assert_allclose(independent,pred[va],rtol=0,atol=1e-7)
   inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20270320+k).split(A,groups=g[tr]));ip=np.full((3,len(tr)),np.nan)
   for j,(it,iv) in enumerate(inner):
    assert m['inner_folds'][j]['train_indices']==tr[it].tolist() and m['inner_folds'][j]['validation_indices']==tr[iv].tolist()
    for ai,alpha in enumerate(s.ALPHAS):ip[ai,iv]=manual(A[it],y[tr[it]],A[iv],alpha)
   np.testing.assert_allclose(ip,m['inner_oof'],rtol=0,atol=1e-7);mse=np.mean((ip-y[tr])**2,axis=1);assert int(np.argmin(mse))==m['selected_index'];assert s.ALPHAS[int(np.argmin(mse))]==m['alpha']
   if k==0 and split=='original':
    perturbed=y.copy();perturbed[va]+=12345.;again=s.fit_arm(X[:,:dim],perturbed,g,tr,k)
    np.testing.assert_array_equal(again['inner_oof'],m['inner_oof']);np.testing.assert_array_equal(again['model'].named_steps['regressor'].dual_coef_,kr.dual_coef_);assert again['alpha']==m['alpha'];replay[arm]={'outer_validation_labels_perturbed':True,'all_inner_choices_and_full_outer_fit_identical':True}
   models+=1
  np.testing.assert_array_equal(arr[arm+'_blend'],.9*ref+.1*pred)
  for name in [arm,arm+'_blend']:assert abs(s.rmse(y,arr[name])-r['scores'][name]['rmse'])<1e-12
 comp=s.compare(y,ref,arr['order2_blend'],fid);assert comp==r['comparisons_vs011']['order2'];mask=~np.isin(ids,s.FIXED5);gain=s.rmse(y[mask],ref[mask])-s.rmse(y[mask],arr['order2_blend'][mask]);assert gain==r['fixed395_gain_vs011']
 gate=comp['gain']>=.02 and comp['positive_folds']>=3 and s.rmse(y,arr['order2'])<s.rmse(y,arr['order1']) and s.rmse(y,arr['order2_blend'])<s.rmse(y,arr['order1_blend']) and gain>0
 assert bool(gate)==r['advance']
 feature_error=None
 if split=='original':
  sc=s.Scattering2D(J=6,shape=(256,256),L=8,max_order=2);feature_error=0.
  for i in [0,199,399]:
   p=_repo_path(ROOT, 'data/train', f'{ids[i]}.png');assert s.sha(p)==ex['image_sha256'][ids[i]]
   with Image.open(_repo_io(p)) as im:x=np.asarray(im,dtype=np.float32)/255.
   a=sc(x).mean((-2,-1),dtype=np.float64);feature_error=max(feature_error,float(abs(a-X[i]).max()));np.testing.assert_array_equal(a,X[i])
  for ident,h in ex['image_sha256'].items():assert s.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==h
 s.check();out={'passed':True,'split':split,'models_verified':models,'reload_error':0.,'independent_kernel_solve_max_error':max(errors),'inner_selection_recomputed':True,'foldtrain_scalers_targets_ids_verified':True,'leakage_replay':replay,'raw_image_feature_repeat_max_error':feature_error,'gate_recomputed':bool(gate),'report_sha256':s.sha(s.OUT/split/'report.json'),'predictions_sha256':s.sha(path),'seconds':time.perf_counter()-start,'holdout_test_access':False,'cost':0,'source_sha256':s.sha(Path(__file__))};s.write(_repo_path(ROOT, f'reports/wavelet-scattering-096-{split}-validation.json'),out);s.emit(out)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=s.SPLITS);run(ap.parse_args().split)
