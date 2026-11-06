import sys,time,gc,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
from common import P,R,np,torch
c.check();assert c.read(R/'compatibility.json')['passed'];old=c.load(P/'experiments/spatial-moments-191/resnet50/features.npz');assert old['ids'].tolist()==c.plan['IDs'] and old['mean'].shape==(400,2048)
smoke={};input_hashes={};images=[]
for ID in c.plan['smoke_IDs']:
 x=c.image_input(ID);images.append(x);input_hashes[ID]=hashlib.sha256(x.numpy().tobytes()).hexdigest()
with c.Monitor('smoke',300) as mon:
 for arm in c.plan['arms']:
  m=c.model(arm);original=c.statehash(m);outputs={};timings={}
  for device in ['cpu','mps']:
   m.to(device);rows=[];seconds=[]
   for x in images:
    assert hashlib.sha256(x.numpy().tobytes()).hexdigest() in input_hashes.values()
    st=time.monotonic()
    with torch.inference_mode():p=c.descriptor(m,x.to(device)).cpu().numpy()[0]
    seconds.append(time.monotonic()-st);assert np.isfinite(p).all();rows.append(p)
   outputs[device]=np.stack(rows);timings[device]=seconds;mon.sample()
  with torch.inference_mode():repeat=np.stack([c.descriptor(m,x.to('mps')).cpu().numpy()[0] for x in images])
  np.testing.assert_array_equal(repeat,outputs['mps']);err=float(abs(outputs['cpu']-outputs['mps']).max());relative=float(np.linalg.norm(outputs['cpu']-outputs['mps'])/max(np.linalg.norm(outputs['cpu']),1e-12));assert err<.001 and relative<.001
  if arm=='imagenet':
   for j,ID in enumerate(c.plan['smoke_IDs']):np.testing.assert_array_equal(outputs['mps'][j],old['mean'][c.plan['IDs'].index(ID)])
  assert original==c.statehash(m);projected=400*float(np.median(timings['mps']));assert projected<600
  c.arrays(R/(arm+'-smoke.npz'),ids=np.array(c.plan['smoke_IDs']),**outputs)
  smoke[arm]={'feature_shape':[3,2048],'CPU_MPS_maxabs':err,'CPU_MPS_relative_L2':relative,'MPS_repeat_exact':True,'state_unchanged':True,'MPS_projected400_seconds':projected,'times':timings,'imagenet_cache_exact':arm=='imagenet'}
  del m;gc.collect();torch.mps.empty_cache();c.emit({'smoke':arm,**smoke[arm]})
 c.write(R/'smoke.json',{'passed':True,'input_tensor_hashes_shared_both_arms':input_hashes,'models':smoke,'no_fit':True})
with c.Monitor('features',600) as mon:
 m=c.model('micronet').to('mps');initial=c.statehash(m);rows=[];image_sha={};start=time.monotonic()
 prior=c.read(P/'experiments/cnn-resolution-188/resnet50/extraction.json')['image_hashes']
 for j,ID in enumerate(c.plan['IDs']):
  x=c.image_input(ID)
  with torch.inference_mode():p=c.descriptor(m,x.to('mps')).cpu().numpy()[0]
  assert np.isfinite(p).all();rows.append(p);image_sha[ID]=c.sha(P/'data/train'/f'{ID}.png');assert image_sha[ID]==prior[ID]
  if (j+1)%100==0:c.emit({'features_done':j+1,'seconds':time.monotonic()-start});mon.sample()
 X=np.stack(rows);assert X.shape==(400,2048) and initial==c.statehash(m)
 sm=c.load(R/'micronet-smoke.npz')
 for j,ID in enumerate(c.plan['smoke_IDs']):np.testing.assert_array_equal(X[c.plan['IDs'].index(ID)],sm['mps'][j])
 c.arrays(R/'features.npz',ids=old['ids'],imagenet=old['mean'],micronet=X)
 c.write(R/'features.json',{'passed':True,'shape_per_arm':[400,2048],'image_sha256':image_sha,'features_SHA256':c.sha(R/'features.npz'),'imagenet_cached_mean_reused':True,'micronet_fresh400':True,'CPU_MPS_smoke_passed':True,'weights_unchanged':True,'state_hash_micronet':initial,'seconds':time.monotonic()-start,'new_gradient_training':0,'no_holdout_test':True})
 del m;gc.collect();torch.mps.empty_cache()
c.check();c.emit({'features_completed':True,'Ridge_training_started':False})
