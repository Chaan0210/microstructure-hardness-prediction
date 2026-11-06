import common as c,engine as e
from common import R,np
c.check();d=c.data();records=[];pred={a:np.full(400,np.nan) for a in ['last5_uniform','update_ema']}
with c.Monitor('training',1200):
 for k in range(5):
  T,V=c.indices(k);p=dict(d);p['y']=d['y'].copy();p['y'][V]=np.nan;assert c.norm(d,T)==c.norm(p,T);out,rec=e.fit(d,k);records.append(rec)
  for arm,values in out.items():pred[arm][V]=values
 assert all(np.isfinite(p).all() for p in pred.values());c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],**pred);c.write(R/'training-complete.json',{'fits':5,'gradient_updates':4000,'shadow_optimizer_verification_steps':4000,'EMA_updates':4000,'all5_train_normalization_poison_exact':True,'folds':records,'OOF_SHA256':c.sha(R/'OOF.npz'),'performance_scores_computed':False,'plan_SHA256':c.sha(R/'plan.json')})
c.check()
