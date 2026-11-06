import common as c,engine as e
from common import R,C,np
c.check();assert c.read(R/'control-validation.json')['passed'];X,cache,d=c.inputs();pred=np.full(400,np.nan);records=[];recipe={'name':'fixed_adamw_low','optimizer':'AdamW','head_lr':.0001,'tail_lr':.00001}
with c.Monitor('training',900):
 for k in range(5):
  T,V=c.indices(k);out=R/f'fold-{k}';out.mkdir();warm=e.ck(C/'convnext_tiny'/f'fold-{k}'/'initial-head.pt');pp,rec=e.train(warm,cache,X,d['y'][T].copy(),recipe,202710186+100*k+99,out,V,ratio=c.plan['ratio'],fold=k);pred[V]=pp;records.append(rec)
  if k==0:
   a=c.load(out/'rng-and-groups.npz');b=c.load(R/'control-fold-0/rng-and-groups.npz')
   for name in ['order','rng_CPU','rng_MPS','decay_factor']:np.testing.assert_array_equal(a[name],b[name])
 assert np.isfinite(pred).all();c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],best021=d['blend_L_MLP'],prediction=pred);c.write(R/'training-complete.json',{'candidate_fits':5,'control_fits':1,'gradient_updates':1440,'candidate_updates':1200,'head_refits':0,'ratio':c.plan['ratio'],'records':records,'OOF_SHA256':c.sha(R/'OOF.npz'),'candidate_OOF_scores_computed':False})
c.check()
