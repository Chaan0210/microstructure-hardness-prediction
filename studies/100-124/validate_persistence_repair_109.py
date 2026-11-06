"""Independent repaired scaling, primal/dual solves, innerCV and leakage checks."""
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
import persistence_repair_109 as s

def manual(X,y,Z,alpha):
 n=len(X);mu=np.sum(X,axis=0)/n;centered=X-mu;variance=np.sum(centered*centered,axis=0)/n;floor=np.sqrt(np.sum(variance)/X.shape[1]);scale=np.maximum(np.sqrt(variance),floor) if floor>0 else np.ones(X.shape[1]);A=centered/scale;B=(Z-mu)/scale;am=A.mean(0);A=A-am;B=B-am;t=(y-y.mean())/y.std();tm=t.mean();dual=np.linalg.solve(A@A.T+alpha*np.eye(n),t-tm);pred=(B@A.T@dual+tm)*y.std()+y.mean();return pred,mu,variance,float(floor),scale

def main(split):
 start=time.perf_counter();plan=s.check();d=s.old.inputs(split);r=s.read(s.OUT/split/'report.json');p=s.OUT/split/'development_predictions.npz';assert s.sha(p)==r['predictions_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 np.testing.assert_array_equal(arr['ids'],d['ids']);np.testing.assert_array_equal(arr['y'],d['y']);np.testing.assert_array_equal(arr['fold'],d['fid']);np.testing.assert_array_equal(arr['reference011'],d['ref']);maxerr=0.;innererr=0.;replay=[];conditionerr=0.;pre=s.read(s.OUT/'preflight.json');assert s.sha(s.OUT/'preflight.json')==r['preflight_sha256'];assert pre['complete'] and not pre['validation_RMSE_computed']
 for arm in s.ARMS:
  X=d[arm]
  for k in range(5):
   tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);p=s.OUT/split/f'{arm}-fold{k}.joblib';assert s.sha(p)==r['models_sha256'][p.name];saved=joblib.load(_repo_io(p));m=saved['model'];assert saved['train_ids']==[d['ids'][i] for i in tr] and saved['validation_ids']==[d['ids'][i] for i in va];assert m.scaler.n_samples_seen_==len(tr);q=m.predict(X[va]);np.testing.assert_array_equal(q,arr[arm][va]);ind,mu,var,floor,scale=manual(X[tr],d['y'][tr],X[va],m.alpha);maxerr=max(maxerr,float(abs(ind-q).max()));np.testing.assert_allclose(ind,q,atol=1e-8,rtol=0);np.testing.assert_array_equal(mu,m.scaler.mean_);np.testing.assert_array_equal(var,m.scaler.var_);assert floor==m.scaler.floor_;np.testing.assert_array_equal(scale,m.scaler.scale_);assert m.ym==d['y'][tr].mean() and m.ys==d['y'][tr].std();assert np.max(m.scaler.transform(X[tr]).var(0))<=1+1e-12
   inner=list(GroupKFold(3,shuffle=True,random_state=20271007+k).split(X[tr],groups=d['g'][tr]));assert saved['detail']['inner_indices']==[[tr[a].tolist(),tr[b].tolist()] for a,b in inner];scores=[]
   for alpha in s.ALPHAS:
    ip=np.full(len(tr),np.nan)
    for it,iv in inner:
     assert not set(d['g'][tr[it]])&set(d['g'][tr[iv]]);ip[iv]=manual(X[tr[it]],d['y'][tr[it]],X[tr[iv]],alpha)[0]
    scores.append({'alpha':alpha,'rmse':s.rmse(d['y'][tr],ip)})
   for x,y in zip(scores,saved['detail']['inner_scores']):innererr=max(innererr,abs(x['rmse']-y['rmse']));assert abs(x['rmse']-y['rmse'])<1e-8
   assert min(scores,key=lambda v:v['rmse'])['alpha']==m.alpha
   if split=='original':
    old=joblib.load(_repo_io(s.old.OUT/'original'/f'{arm}-fold{k}.joblib'))['model'];fixed=s.Regressor(old.alpha).fit(X[tr],d['y'][tr]);record=next(z for z in pre['records'] if z['arm']==arm and z['fold']==k)
    for name,model in [('original',old),('repaired',fixed)]:
     A=model.scaler.transform(X[tr]);A-=A.mean(0);eig=np.linalg.eigvalsh(A@A.T+model.alpha*np.eye(len(tr)));condition=float(eig[-1]/eig[0]);err=abs(condition-record[name]['regularized_dual_gram_condition']);conditionerr=max(conditionerr,err);assert err<1e-9
    with np.load(_repo_io(s.OUT/'preflight-predictions.npz'),allow_pickle=False) as zz:np.testing.assert_array_equal(zz[f'{arm}_fold{k}_original'],old.predict(X[va]));np.testing.assert_array_equal(zz[f'{arm}_fold{k}_repaired'],fixed.predict(X[va]))
   if split=='original' and k==0:
    dd=dict(d);dd['y']=d['y'].copy();dd['y'][va]+=12345;again,mm,detail=s.nested(X,dd,tr,va,k);np.testing.assert_array_equal(again,q);np.testing.assert_array_equal(mm.ridge.coef_,m.ridge.coef_);assert detail==saved['detail'];replay.append(arm)
  np.testing.assert_array_equal(arr[arm+'_blend'],.9*d['ref']+.1*arr[arm])
 assert s.old.gate(d,arr)==r['gate']
 for key in r['scores']:assert s.rmse(d['y'],arr[key])==r['scores'][key]['rmse']
 # Numerical invariants use no validation data or threshold tuning.
 constant=np.ones((10,512));cc=s.TrainFloorScaler().fit(constant);np.testing.assert_array_equal(cc.transform(constant),np.zeros_like(constant));assert cc.floor_==0
 s.check();out={'passed':True,'split':split,'models_reloaded_exact':10,'independent_dual_prediction_max_error':maxerr,'all_inner_CV_scores_recomputed_max_error':innererr,'independent_eigen_condition_max_error':conditionerr,'fold_training_scaler_floor_target_ids_verified':True,'train_scaled_variances_le1':True,'constant_input_fallback_verified':True,'complete_nested_replay_with_perturbed_outer_validation_labels':replay,'preflight_fixed_alpha_predictions_reproduced':split=='original','report_sha256':s.sha(s.OUT/split/'report.json'),'original107_files_preserved':len(plan['original107_preserved']),'protected011_files_preserved':len(plan['protected011']),'source_sha256':s.sha(Path(__file__)),'seconds':time.perf_counter()-start,'holdout_test_used':False};s.write(_repo_path(ROOT, f'reports/persistence-repair-109-{split}-validation.json'),out);s.emit(out)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=s.SPLITS);main(ap.parse_args().split)
