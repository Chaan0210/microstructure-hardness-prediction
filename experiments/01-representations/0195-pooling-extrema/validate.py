import common as c
from common import np,R,P
import ridge as e
from scipy.optimize import brentq
from scipy.linalg import cho_factor,cho_solve
import math
c.check();d=c.data();f=c.load(R/'features.npz');r=c.read(R/'report.json');oof=c.load(R/'OOF.npz');bootstrap=c.load(R/'bootstrap.npz');ex=c.read(R/'extraction.json');assert c.sha(R/'features.npz')==ex['features_SHA256'];assert c.sha(R/'OOF.npz')==r['OOF_SHA256'] and c.sha(R/'bootstrap.npz')==r['bootstrap_SHA256'];src=c.load(P/'artifacts/dinob-062/development_features.npz')
np.testing.assert_array_equal(f['baseline'],src['X']);np.testing.assert_array_equal(f['ids'],d['ids'])
for name in ['ids','y','fold','groups','best021']:np.testing.assert_array_equal(oof[name],d[name])
records=[]
def scalar_rmse(y,p):return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(y,p))/len(y))
with c.Monitor('validation') as monitor:
 for i,a in c.tokens():
  ordered=np.sort(a,axis=0)
  for name,j in [('minimum',0),('second_minimum',1),('second_maximum',-2),('maximum',-1)]:np.testing.assert_array_equal(f[name][i],ordered[j])
  np.testing.assert_array_equal(f['final_std64'][i],a.astype(float).std(0))
 c.emit({'independent_sorted_extrema_all400_exact':True})
 for k in range(5):
  T=np.array(c.plan['folds'][str(k)]['train']);V=np.array(c.plan['folds'][str(k)]['validation']);root=R/f'fold-{k}';base=None
  for arm in c.ARMS:
   X=c.features(arm,f);model=c.load(root/(arm+'-model.npz'));rec=c.read(root/(arm+'-report.json'));saved=c.load(root/(arm+'-predictions.npz'));assert c.sha(root/(arm+'-model.npz'))==rec['model_SHA256'] and c.sha(root/(arm+'-predictions.npz'))==rec['prediction_SHA256'];np.testing.assert_array_equal(saved['indices'],V);np.testing.assert_array_equal(saved['train_indices'],T);np.testing.assert_array_equal(saved['ids'],d['ids'][V]);np.testing.assert_array_equal(model['train'],T)
   Z=e.transform(X,model);p=e.predict(Z,model);np.testing.assert_array_equal(p[V],oof[arm][V]);np.testing.assert_array_equal(p[V],saved['prediction']);np.testing.assert_array_equal(p[T],saved['train_prediction']);assert np.isfinite(p).all()
   xp=X.copy();xp[V]=1e7-1e5*xp[V];yp=d['y'].copy();yp[V]=1e6-1000*yp[V];g,zp,U,Vt=e.geometry(xp,T);mp=e.fit(g,U,Vt,yp,T)
   for key in model:np.testing.assert_array_equal(model[key],mp[key],err_msg=arm+key)
   np.testing.assert_array_equal(zp[T],Z[T]);np.testing.assert_array_equal(e.predict(Z,mp),p);np.testing.assert_allclose((U*g['singular_values'])@Vt,Z[T],atol=1e-10,rtol=1e-10)
   perm=d['y'].copy();perm[T]=perm[T][::-1];pm=e.fit(g,U,Vt,perm,T);assert float(pm['alpha'])==float(model['alpha']) and abs(float(pm['EDF'])-32)<1e-10
   means=[];scales=[]
   for j in range(X.shape[1]//768):
    q=X[T,j*768:(j+1)*768];mu=np.mean(q,axis=0);var=np.mean((q-mu)**2,axis=0);floor=math.sqrt(float(np.mean(var)));scale=np.maximum(np.sqrt(var),floor) if floor else np.ones_like(var);means.append(mu);scales.append(scale)
   mu=np.concatenate(means);scale=np.concatenate(scales);A=(X-mu)/scale;center=A[T].mean(0);A=A-center;np.testing.assert_array_equal(mu,model['mean']);np.testing.assert_array_equal(scale,model['scale']);np.testing.assert_array_equal(center,model['center']);np.testing.assert_array_equal(A,Z)
   if arm=='baseline':base={key:model[key].copy() for key in ['mean','variance','scale','center','bank_RMS']}
   else:
    for key,val in base.items():np.testing.assert_array_equal(model[key][:len(val)],val)
   alpha=float(model['alpha']);K=A[T]@A[T].T;ym=float(d['y'][T].mean());ys=float(d['y'][T].std());cf=cho_factor(K+alpha*np.eye(len(T)),lower=True);weights=cho_solve(cf,(d['y'][T]-ym)/ys);ind=(A@A[T].T)@weights*ys+ym;err=float(abs(ind-p).max());assert err<1e-8
   trace=float(np.trace(cho_solve(cf,K)));assert abs(trace-32)<1e-8;lam=np.maximum(np.linalg.eigvalsh(K),0);upper=float(lam.max())
   while float(np.sum(lam/(lam+upper)))>32:upper*=2
   aa=brentq(lambda t:float(np.sum(lam/(lam+t)))-32,upper*1e-15,upper,xtol=1e-12,rtol=1e-14);alphaerr=abs(aa-alpha)/alpha;assert alphaerr<1e-10
   for y,q,expected in [(d['y'][T],p[T],rec['train_RMSE']),(d['y'][V],p[V],rec['validation_RMSE'])]:assert abs(scalar_rmse(y,q)-expected)<1e-12
   if arm!='baseline':
    Xp=X.copy();Xp[:,-768:]=f['second_maximum' if arm=='plus_max' else 'second_minimum'];delta=e.predict(e.transform(Xp,model),model)[V]-p[V];np.testing.assert_array_equal(delta,saved['second_extreme_prediction_delta'])
   row={'fold':k,'arm':arm,'CPU_reload_ID_order_finite_exact':True,'heldout_X_y_poison_refit_exact':True,'train_statistics_independent_and_shared_baseline_exact':True,'alpha_unchanged_by_permuted_training_labels':True,'independent_dual_prediction_maxabs':err,'independent_Gram_trace_EDF':trace,'independent_Gram_eigenvalues_Brent_alpha_relative_error':alphaerr};records.append(row);c.emit(row);monitor.sample();del X,xp,Z,A,U,Vt,g,model,mp,pm,K
 ps={'best021':d['best021'],**{arm:oof[arm] for arm in c.ARMS}};ps.update({arm+'_blend':.9*d['best021']+.1*oof[arm] for arm in c.ARMS});unique,inv=np.unique(d['groups'],return_inverse=True);np.testing.assert_array_equal(bootstrap['draws'],np.random.default_rng(202710195).integers(len(unique),size=(3000,len(unique))));assert len(unique)==400;row_for_group=np.argsort(inv);ix=row_for_group[bootstrap['draws']];max_boot_error=0.
 for name,score in r['scores'].items():
  assert abs(scalar_rmse(d['y'],ps[name])-score['RMSE'])<1e-12
  for k in range(5):
   v=d['fold']==k;assert abs(scalar_rmse(d['y'][v],ps[name][v])-score['fold_RMSE'][k])<1e-12
 for name,q in r['comparisons'].items():
  y=d['y'];a=ps[q['reference']];b=ps[q['candidate']];gain=scalar_rmse(y,a)-scalar_rmse(y,b);assert abs(gain-q['gain'])<1e-12;bb=np.sqrt(np.mean((y[ix]-a[ix])**2,axis=1))-np.sqrt(np.mean((y[ix]-b[ix])**2,axis=1));error=float(abs(bb-bootstrap[name]).max());assert error<1e-12;max_boot_error=max(max_boot_error,error);np.testing.assert_allclose(np.quantile(bb,[.025,.975]),q['conditional_paired_groupbootstrap_CI95'],rtol=0,atol=1e-12);np.testing.assert_allclose(np.quantile(bb,[.0125,.9875]),q['conditional_CI97_5_sensitivity'],rtol=0,atol=1e-12)
  benefits=[(float(y[i])-float(a[i]))**2-(float(y[i])-float(b[i]))**2 for i in range(400)];order=sorted(range(400),key=lambda i:(-benefits[i],str(d['ids'][i])));assert d['ids'][order[:5]].tolist()==q['top5_removed_IDs'];keep=np.array(order[5:]);assert abs(scalar_rmse(y[keep],a[keep])-scalar_rmse(y[keep],b[keep])-q['top5_removed_gain'])<1e-12;foldgain=[]
  for k in range(5):
   v=d['fold']==k;foldgain.append(scalar_rmse(y[v],a[v])-scalar_rmse(y[v],b[v]))
  np.testing.assert_allclose(foldgain,q['fold_gains'],rtol=0,atol=1e-12);assert sum(z>0 for z in foldgain)==q['positive_folds']
 for arm in ['plus_max','plus_min']:
  q=r['comparisons'][arm+'_blend_vs_best021'];b=r['comparisons'][arm+'_blend_vs_baseline'];a=r['comparisons'][arm+'_standalone_vs_baseline'];gate=a['gain']>0 and b['gain']>0 and b['positive_folds']>=3 and b['top5_removed_gain']>0 and q['gain']>=.02 and q['positive_folds']>=3 and q['conditional_paired_groupbootstrap_CI95'][0]>0 and q['top5_removed_gain']>0;assert bool(gate)==r['candidate_gates'][arm]['numeric_gate_passed']
 assert r['numeric_gate_passed']==any(q['numeric_gate_passed'] for q in r['candidate_gates'].values());assert not c.network_attempts
result={'passed':True,'models_verified':len(records),'all400_sorted_extrema_exact':True,'baseline_all_CLS_mean_std_exact':True,'heldout_poison_full_recipe_refits':15,'trainlabel_permutation_alpha_checks':15,'independent_dual_and_Gram_solvers':15,'checks':records,'max_independent_prediction_error':max(q['independent_dual_prediction_maxabs'] for q in records),'max_independent_EDF_error':max(abs(q['independent_Gram_trace_EDF']-32) for q in records),'max_direct_bootstrap_difference':max_boot_error,'all_scalar_metrics_top5_gate_checks_passed':True,'network_attempts':len(c.network_attempts),'raw_data_access':False};c.check();c.write(R/'validation.json',result);c.emit({key:value for key,value in result.items() if key!='checks'})
