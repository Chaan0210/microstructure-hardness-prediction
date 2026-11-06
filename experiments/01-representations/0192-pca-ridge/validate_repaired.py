import common as c
from common import R,P,np
import ridge as e
from scoring import evaluate
from sklearn.linear_model import Ridge
from scipy.optimize import brentq
import time
c.check();X,d=c.data();OOF=c.load(R/'OOF.npz');report=c.read(R/'report.json');assert c.sha(R/'OOF.npz')==report['OOF_SHA256'];records=[];started=time.perf_counter()
for key,ref in [('ids','ids'),('y','y'),('groups','g'),('fold','fold'),('best021','blend_L_MLP')]:np.testing.assert_array_equal(OOF[key],d[ref])
with c.Monitor('validation-repaired',600) as mon:
 for k in range(5):
  T=np.array(c.plan['folds'][str(k)]['train']);V=np.array(c.plan['folds'][str(k)]['validation']);assert not set(d['g'][T])&set(d['g'][V]);root=R/f'fold-{k}';g=c.load(root/'geometry.npz');Z=e.transform(X,g);xp=X.copy();xp[V]=1e7-1e5*xp[V];yp=d['y'].copy();yp[V]=1e6-1000*yp[V];gp,zp=e.geometry(xp,T)
  for key in g:np.testing.assert_array_equal(g[key],gp[key],err_msg=key)
  np.testing.assert_array_equal(zp[T],Z[T]);np.testing.assert_array_equal(g['train'],T);np.testing.assert_allclose(g['U']@np.diag(g['singular_values'])@g['V'],Z[T],atol=1e-10,rtol=1e-10)
  prior=c.load(P/f'experiments/spatial-moments-191/convnext_tiny/fold-{k}/mean_std/preprocess.npz')
  for j in range(2):
   for a,b in [('raw_mean','mean'),('raw_var','var'),('raw_scale','scale')]:np.testing.assert_array_equal(g[a][j*768:(j+1)*768],prior[f'bank{j}_{b}'])
  C=prior['pca_components'];H=g['V'][:128];subspace_error=float(abs(C-(C@H.T)@H).max());assert subspace_error<1e-8;np.testing.assert_allclose(g['singular_values'][:128],prior['pca_singular_values'],rtol=1e-10,atol=1e-10)
  for arm in c.plan['arms']:
   o=c.load(root/(arm+'.npz'));rec=c.read(root/(arm+'-report.json'));assert c.sha(root/(arm+'.npz'))==rec['model_SHA256'];saved=c.load(root/(arm+'-predictions.npz'));np.testing.assert_array_equal(saved['indices'],V);np.testing.assert_array_equal(saved['ids'],d['ids'][V]);pp=e.predict(Z,o);np.testing.assert_array_equal(pp[V],saved['prediction']);np.testing.assert_array_equal(pp[V],OOF[arm][V]);np.testing.assert_array_equal(pp[T],saved['train_prediction']);assert np.isfinite(pp).all()
   op=e.fit(gp,yp,T,arm)
   for key in o:np.testing.assert_array_equal(op[key],o[key],err_msg=arm+key)
   np.testing.assert_array_equal(e.predict(Z,op),pp);permuted=d['y'].copy();permuted[T]=permuted[T][::-1];operm=e.fit(g,permuted,T,arm);assert float(operm['alpha'])==float(o['alpha']) and float(operm['effective_df'])==float(o['effective_df']);assert abs(float(operm['effective_df'])-32.)<1e-10
   A=Z@g['V'][:128].T if arm=='pca128_ridge' else Z;alpha=float(o['alpha']);yt=(d['y'][T]-float(o['target_mean']))/float(o['target_std']);m=Ridge(alpha=alpha,fit_intercept=False,solver='cholesky');m.fit(A[T],yt);ind=m.predict(A)*float(o['target_std'])+float(o['target_mean']);err=float(abs(ind-pp).max());assert err<1e-8
   K=A[T]@A[T].T;trace=float(np.trace(np.linalg.solve(K+alpha*np.eye(len(T)),K)));assert abs(trace-32)<1e-8;lam=g['singular_values'][:128 if arm=='pca128_ridge' else 320]**2;hi=float(lam.max())
   while np.sum(lam/(lam+hi))>32:hi*=2
   independent_alpha=brentq(lambda a:float(np.sum(lam/(lam+a)))-32,float(lam.max())*1e-15,hi,xtol=1e-12,rtol=1e-14);assert abs(independent_alpha-alpha)/alpha<1e-12
   trainrmse=float(np.sqrt(np.mean((d['y'][T]-pp[T])**2)));assert trainrmse==rec['train_RMSE'];records.append({'fold':k,'arm':arm,'CPU_reload_ID_order_finite_exact':True,'heldout_X_y_poison_geometry_coef_prediction_exact':True,'alpha_invariant_to_changed_training_labels':True,'independent_sklearn_prediction_maxabs':err,'direct_smoother_trace_df':trace,'independent_alpha_relative_error':abs(independent_alpha-alpha)/alpha,'same191_raw_scaler_exact':True,'same191_PCA128_subspace_maxabs':subspace_error});c.emit(records[-1])
  mon.sample()
 for arm in c.plan['arms']:assert evaluate(d,OOF[arm])==report['scores'][arm]
 c.check()
z={'passed':True,'initial_validator_exact_integer_assertion_failure_preserved':True,'repair_record':'validation-assertion-repair.json','additional_partial_first_audit':{'geometry_poison_refits':2,'poison_Ridge_fits':3,'training_label_permutations':3,'independent_sklearn_fits':2},'all10_models_verified':len(records)==10,'checks':records,'max_independent_prediction_error':max(q['independent_sklearn_prediction_maxabs'] for q in records),'max_df_error':max(abs(q['direct_smoother_trace_df']-32) for q in records),'training_label_independent_alpha':True,'full_geometry_heldout_poison_refits':5,'heldout_poison_Ridge_refits':10,'training_label_permutation_checks':10,'independent_sklearn_fits':10,'seconds':time.perf_counter()-started,'network_attempts':len(c.network_attempts),'no_raw_image_access':True};assert not c.network_attempts;c.write(R/'validation.json',z);c.emit({k:v for k,v in z.items() if k!='checks'})
