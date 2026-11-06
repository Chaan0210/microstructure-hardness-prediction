import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
from common import P,R,np
import ridge as e
from sklearn.linear_model import Ridge
from scipy.optimize import brentq
c.check();assert c.read(R/'features.json')['passed'];bank=c.load(R/'features.npz');d=c.data();assert bank['ids'].tolist()==d['ids'].tolist();OOF={a:np.full(400,np.nan) for a in c.plan['arms']};records=[];start=time.monotonic()
c.write(R/'training-started.json',{'started':True,'stage':'Original5fold frozen features Ridge','primary_fits_planned':10,'gradient_training':0,'criteria_SHA256':c.sha(R/'plan.json'),'features_SHA256':c.sha(R/'features.npz'),'code_lock_SHA256':c.sha(R/'code-lock.json')})
with c.Monitor('Ridge-and-validation',600) as mon:
 for arm in c.plan['arms']:
  X=bank[arm].astype(np.float64)
  for k in range(5):
   T=np.array(c.plan['folds'][str(k)]['train']);V=np.array(c.plan['folds'][str(k)]['validation']);path=R/f'{arm}-fold-{k}';path.mkdir();g=e.geometry(X,T);m=e.fit(g,d['y'],T);p=e.predict(X,g,m);assert np.isfinite(p).all();OOF[arm][V]=p[V]
   c.arrays(path/'geometry.npz',**g);c.arrays(path/'model.npz',**m);c.arrays(path/'prediction.npz',ID=d['ids'][V],indices=V,prediction=p[V])
   gr=c.load(path/'geometry.npz');mr=c.load(path/'model.npz');np.testing.assert_array_equal(e.predict(X,gr,mr),p)
   # One held-out poison replay per arm/fold; neither NaNs nor finite poison may enter fit.
   Xp=X.copy();yp=d['y'].copy();Xp[V[::2]]=np.nan;Xp[V[1::2]]=1e8-100*Xp[V[1::2]];yp[V[::2]]=np.nan;yp[V[1::2]]=1e7+yp[V[1::2]]
   gp=e.geometry(Xp,T);mp=e.fit(gp,yp,T)
   for key in g:np.testing.assert_array_equal(g[key],gp[key],err_msg=key)
   for key in m:np.testing.assert_array_equal(m[key],mp[key],err_msg=key)
   np.testing.assert_array_equal(e.predict(X,gp,mp),p)
   alt=d['y'].copy();alt[T]=alt[T][::-1];permuted=e.fit(g,alt,T);assert float(permuted['alpha'])==float(m['alpha'])
   # Independent scaling, Gram eigenvalues, alpha root and solver.
   manual_mu=np.sum(X[T],axis=0)/len(T);manual_var=np.sum((X[T]-manual_mu)**2,axis=0)/len(T);sc=np.maximum(np.sqrt(manual_var),np.sqrt(np.mean(manual_var)));A=(X-manual_mu)/sc;A-=A[T].mean(0)
   np.testing.assert_allclose(A,e.transform(X,g),rtol=1e-12,atol=1e-12)
   K=A[T]@A[T].T;ev=np.maximum(np.linalg.eigvalsh(K),0);hi=float(ev.max())
   while np.sum(ev/(ev+hi))>32:hi*=2
   ia=brentq(lambda a:float(np.sum(ev/(ev+a)))-32,hi*1e-15,hi,xtol=1e-12,rtol=1e-14);alpha=float(m['alpha']);assert abs(ia-alpha)/alpha<1e-10
   yt=(d['y'][T]-float(m['y_mean']))/float(m['y_std']);ind=Ridge(alpha=ia,fit_intercept=False,solver='cholesky').fit(A[T],yt);ip=ind.predict(A)*float(m['y_std'])+float(m['y_mean']);err=float(abs(ip-p).max());assert err<1e-8
   df=float(np.trace(np.linalg.solve(K+alpha*np.eye(len(T)),K)));assert abs(df-32)<1e-8
   assert not set(d['g'][T])&set(d['g'][V]);rec={'arm':arm,'fold':k,'train_n':len(T),'validation_n':len(V),'rank':int(g['rank']),'alpha':alpha,'EDF':float(m['df']),'independent_EDF':df,'independent_alpha_relative_error':abs(ia-alpha)/alpha,'independent_prediction_maxabs':err,'reload_exact':True,'heldout_NaN_finite_poison_exact':True,'train_labels_do_not_change_alpha':True,'model_SHA256':c.sha(path/'model.npz'),'geometry_SHA256':c.sha(path/'geometry.npz'),'prediction_SHA256':c.sha(path/'prediction.npz')};records.append(rec);c.write(path/'validation.json',rec);c.emit({'completed_Ridge_fits':len(records),'arm':arm,'fold':k,'validation_passed':True});mon.sample()
 assert all(np.isfinite(p).all() for p in OOF.values());c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],best021=d['blend_L_MLP'],**OOF)
 c.write(R/'validation.json',{'passed':True,'checks':records,'main_Ridge_fits':10,'poison_Ridge_fits':10,'permuted_y_fits':10,'independent_sklearn_fits':10,'gradient_fits':0,'OOF_SHA256':c.sha(R/'OOF.npz'),'max_independent_prediction_error':max(x['independent_prediction_maxabs'] for x in records),'max_EDF_error':max(abs(x['independent_EDF']-32) for x in records),'seconds':time.monotonic()-start,'all_scores_still_uncomputed':True,'selection_note':'All400 predictions and technical validation saved before scoring.'})
c.check();c.emit({'main_Ridge_fits':10,'validation_complete':True,'scores_not_computed':True})
