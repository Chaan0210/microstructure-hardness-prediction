import common as c,engine as e
from common import R,np
c.check();assert c.read(R/'preflight.json')['passed'];records=[];hashes={}
with c.Monitor('training',2400):
 for split in c.SPLITS:
  d=c.data(split);pred={a:np.full(400,np.nan) for a in ['last5_uniform','update_ema']}
  for k in range(5):
   T,V=c.indices(d,k);p=dict(d);p['y']=d['y'].copy();p['y'][V]=np.nan;assert c.norm(d,T)==c.norm(p,T);out,rec=e.fit(d,k);records.append(rec)
   for arm,values in out.items():pred[arm][V]=values
  assert all(np.isfinite(p).all() for p in pred.values());c.arrays(R/split/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],**pred);hashes[split]=c.sha(R/split/'OOF.npz')
 c.write(R/'training-complete.json',{'fits':10,'gradient_updates':8000,'shadow_optimizer_verification_steps':8000,'EMA_updates':8000,'all10_train_normalization_poison_exact':True,'folds':records,'OOF_SHA256':hashes,'performance_scores_computed':False,'plan_SHA256':c.sha(R/'plan.json')})
c.check()
