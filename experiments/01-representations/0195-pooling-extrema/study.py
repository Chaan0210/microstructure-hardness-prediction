import common as c
from common import np,R
import ridge as e
from scoring import rmse,evaluate
c.check();d=c.data();f=c.load(R/'features.npz');assert c.sha(R/'features.npz')==c.read(R/'extraction.json')['features_SHA256'];np.testing.assert_array_equal(d['ids'],f['ids']);pred={a:np.full(400,np.nan) for a in c.ARMS};records=[]
with c.Monitor('study') as monitor:
 for k in range(5):
  T=np.array(c.plan['folds'][str(k)]['train']);V=np.array(c.plan['folds'][str(k)]['validation']);out=R/f'fold-{k}';out.mkdir();baseline_geometry=None
  for arm in c.ARMS:
   X=c.features(arm,f);g,Z,U,Vt=e.geometry(X,T)
   if arm=='baseline':baseline_geometry={n:g[n].copy() for n in ['mean','variance','scale','center','bank_RMS']}
   else:
    for name,val in baseline_geometry.items():np.testing.assert_array_equal(g[name][:len(val)],val)
   model=e.fit(g,U,Vt,d['y'],T);p=e.predict(Z,model);assert np.isfinite(p).all();pred[arm][V]=p[V];c.arrays(out/(arm+'-model.npz'),**model);reloaded=c.load(out/(arm+'-model.npz'));np.testing.assert_array_equal(e.predict(e.transform(X,reloaded),reloaded),p)
   sensitivity=None;delta=np.zeros(len(V))
   if arm!='baseline':
    Xp=X.copy();Xp[:,-768:]=f['second_maximum' if arm=='plus_max' else 'second_minimum'];pp=e.predict(e.transform(Xp,model),model);delta=pp[V]-p[V];order=np.argsort(-abs(delta),kind='stable')[:5];sensitivity={'interpretation':'Channelwise second-extremum probe, no refit/no score-based selection; not a valid physical image perturbation.','absolute_prediction_delta_quantiles':np.quantile(abs(delta),[0,.5,.95,.99,1]).tolist(),'largest_validation_changes':[{'ID':str(d['ids'][V[j]]),'delta_raw_target':float(delta[j])} for j in order]}
   c.arrays(out/(arm+'-predictions.npz'),indices=V,ids=d['ids'][V],prediction=p[V],train_indices=T,train_prediction=p[T],second_extreme_prediction_delta=delta)
   rec={'fold':k,'arm':arm,'dimensions':X.shape[1],'alpha':float(model['alpha']),'EDF_excluding_intercept':float(model['EDF']),'train_RMSE':rmse(d['y'][T],p[T]),'validation_RMSE':rmse(d['y'][V],p[V]),'spectral_and_scaling_diagnostics':e.diagnostics(g,Z,U,float(model['alpha']),T,V),'extreme_sensitivity':sensitivity,'CPU_reload_exact':True,'model_SHA256':c.sha(out/(arm+'-model.npz')),'prediction_SHA256':c.sha(out/(arm+'-predictions.npz'))};c.write(out/(arm+'-report.json'),rec);records.append(rec);c.emit({n:rec[n] for n in ['fold','arm','dimensions','alpha','EDF_excluding_intercept','train_RMSE','validation_RMSE','CPU_reload_exact']});monitor.sample();del X,g,Z,U,Vt,model,reloaded
 assert all(np.isfinite(x).all() for x in pred.values());r,draws,boot=evaluate(d,pred);c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['groups'],best021=d['best021'],**pred);c.arrays(R/'bootstrap.npz',draws=draws,**boot);r.update({'fold_records':records,'estimand':c.plan['evaluation'],'OOF_SHA256':c.sha(R/'OOF.npz'),'bootstrap_SHA256':c.sha(R/'bootstrap.npz')});c.write(R/'report.json',r);c.emit({n:r[n] for n in ['scores','comparisons','candidate_gates','numeric_gate_passed']})
c.check()
