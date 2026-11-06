import common as c,engine as e
from common import R,P,np,torch
c.check();assert c.read(R/'structure-preflight.json')['numeric_structure_gate_passed'];assert c.read(R/'visual-review.json')['passed']
with c.Monitor('setup',300):
 d=c.data();T,V=c.indices(0);norm=c.norm(d,T);x=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];rng=np.random.default_rng(20261003);ix=rng.permutation(320)[:32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);xb=np.rot90(x[ix],rot,axes=(2,3));xb=xb[:,:,:,::-1] if flip else xb;xb=np.ascontiguousarray(xb);checks=[]
 for dev in ['cpu']+(['mps'] if c.device=='mps' else []):
  states=[];rstates=[]
  for branch in ['legacy','off']:
   m=e.initialize(0,dev);o=e.optimize(m);a=xb if branch=='legacy' else np.ascontiguousarray(xb.copy());e.step(m,o,torch.from_numpy(a).to(dev),torch.tensor(y[ix],dtype=torch.float32,device=dev));states.append({n:v.detach().cpu().clone() for n,v in m.state_dict().items()});rstates.append(c.rnghash());del m,o
  assert all(torch.equal(states[0][n],v) for n,v in states[1].items());assert rstates[0]==rstates[1];checks.append({'device':dev,'one_update_state_and_RNG_exact':True})
 # Derive diagnostic cuts using train data only, before any model outcomes.
 C=np.quantile(d['images'],[.05,.95],axis=(1,2,3));C=C[1]-C[0]
 with (P/'artifacts/grain-visual-audit-097/all400_noise_contrast.csv').open() as f:rows={r['ID']:r for r in c.csv.DictReader(f)}
 assert set(rows)==set(d['ids']);N=np.array([float(rows[i]['noise'])/max(float(rows[i]['contrast_q95_q05']),1e-12) for i in d['ids']]);out={'contrast':C,'noise_contrast_ratio':N};bound=[]
 for name,q in list(out.items()):
  bins=np.full(400,-1,int)
  for k in range(5):
   T,V=c.indices(k);edges=np.quantile(q[T],[1/3,2/3],method='linear');bins[V]=np.searchsorted(edges,q[V],side='right');bound.append({'proxy':name,'fold':k,'train_tertiles':edges.tolist()})
  out[name+'_bins']=bins
 c.arrays(R/'quality-bins.npz',ids=d['ids'],**out);c.write(R/'quality-bins.json',{'boundaries':bound,'target_used_for_bins':False,'SHA256':c.sha(R/'quality-bins.npz')})
 f=c.load(P/'experiments/matched-replacement-194/frozen-components.npz');old=c.load(P/'artifacts/cnn-seed-mean-133/audit/original.npz');np.testing.assert_array_equal(old['ids'],d['ids']);np.testing.assert_array_equal(old['y'],d['y']);np.testing.assert_array_equal(old['fold'],d['fold']);np.testing.assert_array_equal(f['ids'],d['ids']);np.testing.assert_array_equal(f['best021'],d['blend_L_MLP'])
 for k in range(5):
  T,V=c.indices(k);np.testing.assert_array_equal(f['coef'][V],np.full(len(V),c.plan['fixed_ensemble']['coef_by_fold'][str(k)]));assert c.plan['fixed_ensemble']['coef_by_fold'][str(k)]==.081*c.plan['fixed_ensemble']['calibrators'][str(k)]['slope'];p=dict(d);p['y']=d['y'].copy();p['y'][V]=np.nan;assert c.norm(d,T)==c.norm(p,T)
 assert c.read(P/'experiments/matched-replacement-194/preflight.json')['passed'];assert c.read(P/'experiments/matched-replacement-194/validation.json')['passed']
 c.arrays(R/'frozen-reference.npz',ids=d['ids'],best021=f['best021'],coefficient=f['coef'],old_first_CNN=old['seeds'][0]);c.write(R/'setup-validation.json',{'passed':True,'null_update':checks,'all5_validation_label_poison_normalization_exact':True,'fixed_ensemble_source194_verified':True,'quality_cuts_before_model_scores':True,'new_model_scores_computed':False,'plan_SHA256':c.sha(R/'plan.json'),'frozen_reference_SHA256':c.sha(R/'frozen-reference.npz')})
c.check()
