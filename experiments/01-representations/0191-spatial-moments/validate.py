from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,np,torch
import pipeline as e
import time,gc,math,hashlib

def run():
 c.check();d=c.data('original');checks=[];replays=[];start=time.monotonic()
 for n,h in c.read(R/'validation-code-lock.json').items():assert c.sha(R/n)==h
 with c.Monitor('validation',900):
  for name in c.plan['models']:
   root=R/name;OOF=c.load(root/'OOF.npz');report=c.read(root/'report.json');assert c.sha(root/'OOF.npz')==report['OOF_SHA256']
   for key,ref in [('ids','ids'),('y','y'),('groups','g'),('fold','fold'),('best021','blend_L_MLP')]:np.testing.assert_array_equal(OOF[key],d[ref])
   for arm in c.plan['arms']:assert e.old.evaluate(d,OOF[arm])==report['scores'][arm]
   width=2048 if name=='resnet50' else 768
   for k in range(5):
    T=np.array(c.plan['folds'][str(k)]['train']);V=np.array(c.plan['folds'][str(k)]['validation']);assert np.array_equal(V,np.flatnonzero(d['fold']==k));assert not set(d['g'][T])&set(d['g'][V]);pair=[]
    for arm in c.plan['arms']:
     out=root/f'fold-{k}'/arm;X=e.raw_features(name,arm);s=c.load(out/'preprocess.npz');o=e.ck(out/'head.pt');rec=c.read(out/'report.json')
     for fn,key in [('head.pt','head_SHA256'),('preprocess.npz','preprocess_SHA256'),('predictions.npz','prediction_SHA256'),('updates.npz','updates_SHA256')]:assert c.sha(out/fn)==rec[key]
     assert o['train']==T.tolist() and o['validation']==V.tolist() and o['seed']==202710185+k and o['dim']==128 and o['parameter_count']==8321;np.testing.assert_array_equal(s['train'],T)
     Z=e.transform(X,s);xp=X.copy();xp[V]=1e7-1e5*xp[V];yp=d['y'].copy();yp[V]=1e6-1000*yp[V];zp,sp=e.fit_pre(xp,T,width)
     for key in s:np.testing.assert_array_equal(s[key],sp[key],err_msg=name+arm+key)
     np.testing.assert_array_equal(Z[T],zp[T]);np.testing.assert_allclose(s['pca_components']@s['pca_components'].T,np.eye(128),atol=1e-10,rtol=1e-10)
     np.testing.assert_allclose(Z[T].var(0,ddof=1),s['pca_explained_variance'],rtol=1e-9,atol=1e-10);expected=e.stats(zp,yp,T,V,o['seed'])
     for key in ['feature_mean','feature_var','feature_scale']:assert torch.equal(o[key],expected[key])
     assert o['target_mean']==expected['target_mean'] and o['target_std']==expected['target_std'];torch.manual_seed(o['seed']);h=e.old.head(128);assert c.statehash(h)==o['initial_head_sha256'];h.load_state_dict(o['head']);assert c.statehash(h)==o['final_head_sha256'];saved=c.load(out/'predictions.npz');np.testing.assert_array_equal(saved['ids'],d['ids'][V]);np.testing.assert_array_equal(saved['indices'],V);np.testing.assert_array_equal(saved['prediction'],OOF[arm][V]);pp=e.predict(h,Z,V,o);np.testing.assert_array_equal(pp,saved['prediction']);h.to('mps');mp=e.predict(h,Z,V,o,'mps');err=float(abs(pp-mp).max());assert err<.002
     u=c.load(out/'updates.npz')['values'];assert u.shape==(500,6) and np.isfinite(u).all();rng=np.random.default_rng(o['seed']+1);orders=[]
     for epoch in range(100):
      orders.append(rng.permutation(len(T)));rows=u[epoch*5:(epoch+1)*5];np.testing.assert_array_equal(rows[:,:3],np.array([[epoch,j,64] for j in range(5)]));lr=1e-5+(.0003-1e-5)*(1+math.cos(math.pi*epoch/100))/2;np.testing.assert_allclose(rows[:,3],lr,rtol=1e-12,atol=1e-16)
     assert hashlib.sha256(np.stack(orders).tobytes()).hexdigest()==o['order_sha256'];assert e.diagnose(X,Z,s,o,T,V)==rec['scale'];pair.append((o,s,u));checks.append({'model':name,'arm':arm,'fold':k,'CPU_reload_exact':True,'CPU_MPS_maxabs':err,'poisoned_validation_X_y_preprocessing_exact':True,'PCA_orthonormal_covariance_verified':True,'train_stats_seed_order_LR_updates_IDs_verified':True});del h;gc.collect();torch.mps.empty_cache()
     if k==0:
      h2,r,uu=e.fit_head(zp,yp,T,V,o['seed'])
      for key in ['head']:
       for n in o[key]:assert torch.equal(r[key][n],o[key][n]),(name,arm,n)
      assert r['curve']==o['curve'] and r['order_sha256']==o['order_sha256'];np.testing.assert_array_equal(uu,u);np.testing.assert_array_equal(e.predict(h2,Z,V,r),pp);audit=root/('audit-poison-'+arm);audit.mkdir();torch.save(r,_repo_io(audit/'head.pt'));c.arrays(audit/'updates.npz',values=uu);replays.append({'model':name,'arm':arm,'fold':0,'full100_epoch_weights_predictions_loss_gradient_LR_orders_exact':True,'poisoned_validation_feature_and_label_refit_train_only':True});c.emit({'poison_replay_passed':name,'arm':arm});del h2
    assert pair[0][0]['initial_head_sha256']==pair[1][0]['initial_head_sha256'] and pair[0][0]['order_sha256']==pair[1][0]['order_sha256'];np.testing.assert_array_equal(pair[0][2][:,:4],pair[1][2][:,:4])
    for key in ['mean','var','scale','RMS']:np.testing.assert_array_equal(pair[0][1]['bank0_'+key],pair[1][1]['bank0_'+key])
    c.emit({'validated':name,'fold':k,'seconds':time.monotonic()-start})
  # Mechanism fixture: equal means do not imply equal std; spatial permutations preserve both.
  a=np.array([-1.,1.,-1.,1.]);b=2*a;assert a.mean()==b.mean() and a.std()!=b.std();assert np.array_equal(np.array([a.mean(),a.std()]),np.array([a[::-1].mean(),a[::-1].std()]))
  c.check()
 result={'passed':True,'heads_verified':len(checks),'checks':checks,'full_poison_replays':replays,'CPU_absolute_tolerance':.002,'CPU_MPS_maxabs':max(x['CPU_MPS_maxabs'] for x in checks),'all_train_only_PCA_scaling_verified':True,'mean_bank_unchanged_across_arms':True,'same_head_capacity_initialization_order_budget_verified':True,'mechanism_fixture_equal_mean_different_std':True,'spatial_permutation_invariant_marginal_statistics':True,'network_attempts':len(c.network_attempts),'seconds':time.monotonic()-start};assert len(checks)==20 and len(replays)==4 and not c.network_attempts;c.write(R/'validation.json',result);c.emit({k:v for k,v in result.items() if k not in ['checks','full_poison_replays']})
if __name__=='__main__':run()
