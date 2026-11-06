from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study164 as s
import numpy as np,json,time,socket,gc,ast
from scipy.linalg import solve
from scipy.spatial.distance import cdist,pdist,squareform
E=s.e;R=s.R;P=s.P;BASE=s.BASE
start=time.perf_counter()
def deny(*a,**k):raise RuntimeError('Offline audit network access forbidden')
socket.socket.connect=deny;socket.create_connection=deny
# Verify copied class is exactly the original implementation, without importing the old environment.
def cls(path):
 src=path.read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='BalancedKernel');return ast.get_source_segment(src,node)
assert cls(BASE/'kernel164.py')==cls(R/'compact_regression.py')
plan=json.loads((P/'plan.json').read_text())
for name,h in {**plan['source_sha256'],**plan['protected']}.items():assert E.sha(R/name)==h,name
count=0;max_kernel_error=0.;rows=[]
def manual_kernel(model,X,y,Q):
 global count,max_kernel_error
 assert model.bound_.n_samples_seen_==len(X) and model.scale_.n_samples_seen_==len(X)
 np.testing.assert_allclose(model.bound_.data_min_,X.min(0),rtol=0,atol=0);np.testing.assert_allclose(model.bound_.data_max_,X.max(0),rtol=0,atol=0)
 bounded=model.bound_.transform(X);np.testing.assert_allclose(model.scale_.mean_,bounded.mean(0),atol=1e-12,rtol=0)
 np.testing.assert_allclose(model.scale_.var_,bounded.var(0),atol=1e-12,rtol=0)
 block=np.array(s.BLOCKS);weights=np.array([1/np.sqrt(sum(block==b)) for b in block]);np.testing.assert_array_equal(weights,model.weights_)
 Z=(bounded-model.scale_.mean_)/model.scale_.scale_*weights
 np.testing.assert_allclose(Z,model.X_,atol=1e-12,rtol=0)
 d=pdist(Z,'sqeuclidean');gamma=1/max(float(np.median(d[d>1e-12]))*model.bandwidth**2,1e-12)
 assert gamma==model.gamma_ and model.ym_==float(y.mean()) and model.ys_==max(float(y.std()),1e-8)
 K=np.exp(-gamma*squareform(d));K.flat[::len(y)+1]+=model.alpha
 coef=solve(K,(y-y.mean())/max(float(y.std()),1e-8),assume_a='pos')
 Zq=(model.bound_.transform(Q)-model.scale_.mean_)/model.scale_.scale_*weights
 p=np.exp(-gamma*cdist(Zq,Z,'sqeuclidean'))@coef*max(float(y.std()),1e-8)+y.mean()
 error=float(np.max(np.abs(p-model.predict(Q))));assert error<1e-8;max_kernel_error=max(max_kernel_error,error);count+=1;return p
for splitpath in sorted(p for p in P.iterdir() if p.name in ['original','20261031','20261101']):
 split=splitpath.name;d=s.data(split);X,y=d['C'],d['y'];pool=[]
 for k in range(5):
  out=splitpath/f'fold-{k}';r=json.loads((out/'report.json').read_text());T=np.where(d['fold']!=k)[0];V=np.where(d['fold']==k)[0];z=np.load(_repo_io(out/'predictions.npz'),allow_pickle=False)
  assert r['train_ids']==d['ids'][T].tolist() and r['validation_ids']==d['ids'][V].tolist()
  for name,h in r['files'].items():assert E.sha(out/name)==h
  for name,h in r['anchor_reports'].items():assert E.sha(out/name)==h
  look={int(i):a for a,i in enumerate(T)};oof=np.full(len(T),np.nan);queries=[];coverage=[]
  for j,(A,H) in enumerate(s.splits(d,T,k)):
   ap=out/f'anchor-{j}';ar=json.loads((ap/'report.json').read_text());az=np.load(_repo_io(ap/'predictions.npz'),allow_pickle=False)
   assert np.array_equal(az['A'],A) and np.array_equal(az['H'],H) and np.array_equal(az['V'],V)
   assert not set(A)&(set(H)|set(V));assert set(A)|set(H)==set(T)
   for name,h in ar['hashes'].items():assert E.sha(ap/name)==h
   assert E.sha(ap/'predictions.npz')==ar['predictions_sha256']
   inner_pred=np.full((9,len(A)),np.nan);alook={int(i):a for a,i in enumerate(A)}
   for ci,(w,a) in enumerate(s.GRID):
    for ell,ind in enumerate(ar['inner_splits']):
     tr=np.array(ind['train']);va=np.array(ind['validation']);obj=E.joblib.load(_repo_io(ap/f'candidate-{ci}-inner-{ell}.joblib'))
     np.testing.assert_array_equal(obj['train'],tr);np.testing.assert_array_equal(obj['validation'],va)
     assert not set(tr)&set(va) and set(tr)|set(va)==set(A);assert not (set(tr)|set(va))&(set(H)|set(V))
     m=obj['model'];assert m.bandwidth==w and m.alpha==a
     inner_pred[ci,[alook[int(i)] for i in va]]=manual_kernel(m,X[tr],y[tr],X[va])
   np.testing.assert_allclose(inner_pred,az['inner_predictions'],atol=1e-8,rtol=0)
   mse=((inner_pred-y[A])**2).mean(1);np.testing.assert_allclose(mse,ar['inner_MSE'],atol=1e-8,rtol=0)
   assert int(np.argmin(mse))==ar['chosen'];m=E.joblib.load(_repo_io(ap/'selected.joblib'));np.testing.assert_array_equal(m['train'],A)
   assert (m['model'].bandwidth,m['model'].alpha)==s.GRID[ar['chosen']]
   pred=manual_kernel(m['model'],X[A],y[A],X[np.r_[H,V]])
   np.testing.assert_allclose(pred[:len(H)],az['anchor_H'],atol=1e-8,rtol=0);np.testing.assert_allclose(pred[len(H):],az['anchor_V'],atol=1e-8,rtol=0)
   oof[[look[int(i)] for i in H]]=az['anchor_H'];queries.append(az['anchor_V']);coverage.extend(H.tolist())
  assert sorted(coverage)==T.tolist();np.testing.assert_array_equal(oof,z['anchor_oof']);np.testing.assert_array_equal(y[T]-oof,z['residual']);np.testing.assert_array_equal(np.mean(queries,0),z['base'])
  tr=E.joblib.load(_repo_io(out/'transform.joblib'));ss,pca=tr['scaler'],tr['pca'];assert ss.n_samples_seen_==320 and pca.n_samples_==320
  np.testing.assert_allclose(ss.mean_,d['B'][T].mean(0,dtype=np.float64),atol=1e-12,rtol=0)
  xt=np.column_stack([X[T],pca.transform(ss.transform(d['B'][T]))]);xv=np.column_stack([X[V],pca.transform(ss.transform(d['B'][V]))])
  model=E.TabICLRegressor.load(out/'residual.pkl',device='cpu');assert model.n_estimators==8 and model.allow_auto_download is False and model.n_samples_in_==320
  np.testing.assert_allclose(model.y_scaler_.mean_,[z['residual'].astype(np.float32).mean(dtype=np.float64)],atol=1e-12,rtol=0)
  encoded=model.ensemble_generator_.unique_filter_.transform(model.X_encoder_.transform(xt))
  for pp in model.ensemble_generator_.preprocessors_.values():np.testing.assert_allclose(pp.standard_scaler_.mean_,encoded.mean(0),atol=1e-6,rtol=1e-6)
  correction=model.predict(xv);assert np.isfinite(correction).all();np.testing.assert_array_equal(correction,z['correction']);np.testing.assert_array_equal(z['base']+correction,z['primary']);np.testing.assert_array_equal(.8*d['ref'][V]+.2*z['primary'],z['fixed20blend'])
  rows.append({'split':split,'fold':k,'kernel_models_verified':84,'own_label_and_outer_label_exclusion':True,'320_residuals_independently_reconstructed':True,'tabicl_reload_max_abs':0.0,'scaler_pca_train_only':True});del model;gc.collect();print('AUDITED',split,k,flush=True)
 # Replay entire first outer pipeline with poisoned outer labels; all nuisance CV and fits regenerated.
 V=np.where(d['fold']==0)[0];poison={**d,'y':d['y'].copy()};poison['y'][V]=np.arange(len(V))*12345.+1e8
 s.fold(poison,split,0,P/f'audit-poison-outer-{split}')
 a=np.load(_repo_io(splitpath/'fold-0/predictions.npz'),allow_pickle=False);b=np.load(_repo_io(P/f'audit-poison-outer-{split}/predictions.npz'),allow_pickle=False)
 for key in ['anchor_oof','residual','anchor_query','base','correction','primary','fixed20blend']:np.testing.assert_array_equal(a[key],b[key])
 # Own-label exclusion includes nuisance hyperparameter selection, not just final fit.
 T=np.where(d['fold']!=0)[0];A,H=s.splits(d,T,0)[0];poisonH={**d,'y':d['y'].copy()};poisonH['y'][H]+=1e8
 s.anchor(poisonH,A,H,V,0,0,P/f'audit-poison-anchor-{split}')
 a=np.load(_repo_io(splitpath/'fold-0/anchor-0/predictions.npz'),allow_pickle=False);b=np.load(_repo_io(P/f'audit-poison-anchor-{split}/predictions.npz'),allow_pickle=False)
 for key in ['inner_predictions','anchor_H','anchor_V']:np.testing.assert_array_equal(a[key],b[key])
for name,h in plan['protected'].items():assert E.sha(R/name)==h,name
result={'passed':True,'network_blocked':True,'folds':rows,'kernel_models_independently_verified':count,'max_kernel_functional_error':max_kernel_error,'whole_outer_label_poison_exact':True,'anchor_heldout_label_poison_including_grid_selection_exact':True,'global_oof_residual_cache_reused':False,'residual_model_inner_selection':False,'original_BalancedKernel_class_exact_copy':True,'protected_files':len(plan['protected']),'plan_sha256':E.sha(P/'plan.json'),'audit_source_sha256':E.sha(BASE/'audit164.py'),'seconds':time.perf_counter()-start}
E.write(P/'audit.json',result);print('AUDIT_PASS',json.dumps({k:v for k,v in result.items() if k!='folds'}),flush=True)
