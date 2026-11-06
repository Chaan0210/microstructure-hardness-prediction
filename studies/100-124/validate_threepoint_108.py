"""Independent kernel algebra/selection, saved-state and feature replay for108."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
import threepoint_study_108 as s
from validate_frame_features_101 import manual

def main(split):
 start=time.perf_counter();p=s.check();d=s.inputs(split);r=s.read(s.OUT/split/'report.json');path=s.OUT/split/'development_predictions.npz';assert s.sha(path)==r['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 np.testing.assert_array_equal(arr['ids'],d['ids']);np.testing.assert_array_equal(arr['y'],d['y']);np.testing.assert_array_equal(arr['fold'],d['fid']);np.testing.assert_array_equal(arr['reference011'],d['ref']);reload=0.;algebra=0.;innererr=0.;replay=[]
 with np.load(_repo_path(s.ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['X'],d['features'][:,:26]);np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['groups'],d['g'])
 for arm,dim in s.DIMS.items():
  X=d['features'][:,:dim];blocks=p['blocks'][:dim]
  for k in range(5):
   path=s.OUT/split/f'{arm}-fold{k}.joblib';assert s.sha(path)==r['models_sha256'][path.name];a=joblib.load(_repo_io(path));tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va];m=a['model'];pred=m.predict(X[va]);np.testing.assert_array_equal(pred,arr[arm][va]);reload=max(reload,float(abs(pred-arr[arm][va]).max()))
   np.testing.assert_array_equal(m.bound_.data_min_,X[tr].min(0));np.testing.assert_array_equal(m.bound_.data_max_,X[tr].max(0));A=m.bound_.transform(X[tr]);np.testing.assert_allclose(m.scale_.mean_,A.mean(0),atol=1e-12,rtol=0);np.testing.assert_allclose(m.scale_.var_,A.var(0),atol=1e-12,rtol=0);assert m.ym_==d['y'][tr].mean() and m.ys_==max(d['y'][tr].std(),1e-8)
   q=manual(X[tr],d['y'][tr],X[va],blocks,m.bandwidth,m.alpha);algebra=max(algebra,float(abs(q-pred).max()));np.testing.assert_allclose(q,pred,atol=1e-7,rtol=0)
   inner=list(GroupKFold(3,shuffle=True,random_state=20261005+k).split(X[tr],groups=d['g'][tr]));scores=[]
   for width in (.25,1.,4.):
    for alpha in (.01,.1,1.):
     ip=np.full(len(tr),np.nan)
     for it,iv in inner:
      assert not set(d['g'][tr[it]])&set(d['g'][tr[iv]]);ip[iv]=manual(X[tr[it]],d['y'][tr[it]],X[tr[iv]],blocks,width,alpha)
     scores.append({'bandwidth':width,'alpha':alpha,'inner_rmse':s.rmse(d['y'][tr],ip)})
   for a1,b1 in zip(scores,a['detail']['inner_scores']):innererr=max(innererr,abs(a1['inner_rmse']-b1['inner_rmse']));assert abs(a1['inner_rmse']-b1['inner_rmse'])<1e-7
   chosen=min(scores,key=lambda v:v['inner_rmse']);assert chosen['bandwidth']==m.bandwidth and chosen['alpha']==m.alpha
   if split=='original' and k==0:
    y=d['y'].copy();y[va]+=12345;q,mm,detail=s.fv.kernel_fold(X,y,d['g'],blocks,tr,va,k);np.testing.assert_array_equal(q,pred);np.testing.assert_array_equal(mm.coef_,m.coef_);assert detail==a['detail'];replay.append(arm)
  np.testing.assert_array_equal(arr[arm+'_blend'],.9*d['ref']+.1*arr[arm])
 assert s.gate(d,arr)==r['gate'];featureerr=0.
 if split=='original':
  for i in (0,199,399):
   expected=np.concatenate(s.f.independent(np.asarray(d['X'][i,0])));featureerr=max(featureerr,float(abs(expected-d['features'][i,26:]).max()));np.testing.assert_allclose(expected,d['features'][i,26:],rtol=0,atol=1e-15)
 s.check();report={'passed':True,'split':split,'models_verified':10,'reload_max_error':reload,'independent_kernel_prediction_max_error':algebra,'independent_inner_RMSE_max_error':innererr,'inner_selections_all_recomputed':True,'outer_train_scaler_target_verified':True,'original_fold0_botharms_innerCV_outerfit_with_perturbed_validation_labels':replay,'independent3_raw_feature_max_error':featureerr,'old26_unchanged':True,'report_sha256':s.sha(s.OUT/split/'report.json'),'source_sha256':s.sha(Path(__file__)),'seconds':time.perf_counter()-start,'holdout_test_used':False};s.write(_repo_path(s.ROOT, f'reports/threepoint-108-{split}-validation.json'),report);s.emit(report)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=s.SPLITS);main(ap.parse_args().split)
