import common as c
from common import R,np
import ridge as e
from scoring import evaluate
import time
c.check();X,d=c.data();preds={a:np.full(400,np.nan) for a in c.plan['arms']};records=[];core=0.
with c.Monitor('study',600) as mon:
 for k in range(5):
  f=c.plan['folds'][str(k)];T=np.array(f['train']);V=np.array(f['validation']);assert np.array_equal(V,np.flatnonzero(d['fold']==k));assert not set(d['g'][T])&set(d['g'][V]);out=R/f'fold-{k}';out.mkdir();start=time.perf_counter();g,Z=e.geometry(X,T);c.arrays(out/'geometry.npz',**g);pair=[]
  for arm in c.plan['arms']:
   o=e.fit(g,d['y'],T,arm);allpred=e.predict(Z,o);assert np.isfinite(allpred).all();pp=allpred[V];preds[arm][V]=pp;c.arrays(out/(arm+'.npz'),**o);q=c.load(out/(arm+'.npz'));np.testing.assert_array_equal(e.predict(e.transform(X,c.load(out/'geometry.npz')),q)[V],pp);c.arrays(out/(arm+'-predictions.npz'),ids=d['ids'][V],indices=V,prediction=pp,train_indices=T,train_prediction=allpred[T]);lam=g['singular_values']**2;df_tail=float(np.sum(lam[128:]/(lam[128:]+float(o['alpha'])))) if arm=='full_ridge' else 0.
   rec={'fold':k,'arm':arm,'alpha':float(o['alpha']),'effective_df_excluding_intercept':float(o['effective_df']),'train_RMSE':float(np.sqrt(np.mean((d['y'][T]-allpred[T])**2))),'validation_RMSE':float(np.sqrt(np.mean((d['y'][V]-pp)**2))),'df_beyond_component128':df_tail,'PCA128_variance_fraction':float(lam[:128].sum()/lam.sum()),'CPU_reload_exact':True,'model_SHA256':c.sha(out/(arm+'.npz')),'predictions_SHA256':c.sha(out/(arm+'-predictions.npz'))};c.write(out/(arm+'-report.json'),rec);records.append(rec);pair.append(rec);c.emit(rec)
  assert pair[1]['alpha']>=pair[0]['alpha'];core+=time.perf_counter()-start;mon.sample()
 assert all(np.isfinite(p).all() for p in preds.values());c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],best021=d['blend_L_MLP'],**preds);report={'scores':{a:evaluate(d,p) for a,p in preds.items()},'fold_records':records,'geometry_ridge_and_reload_core_seconds':core,'OOF_SHA256':c.sha(R/'OOF.npz')};c.write(R/'report.json',report);c.emit({'completed':'192 main5folds','scores':report['scores'],'core_seconds':core})
c.check()
