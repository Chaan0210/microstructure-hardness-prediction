import common as c
from common import R,P,np,torch
import argparse,time,gc

def smoke():
 p=c.check();records={}
 with c.Monitor('smoke',300) as mon:
  for name in p['models']:
   prefix,tail=c.model(name);initial=[c.statehash(x) for x in [prefix,tail]];outputs={};times={}
   for device in ['cpu','mps']:
    prefix.to(device);tail.to(device);out=[];seconds=[]
    for i in p['smoke_IDs']:
     st=time.monotonic()
     with torch.inference_mode():a=tail(prefix(c.image_input(i).to(device))).cpu().numpy()[0]
     assert np.isfinite(a).all();out.append(a);seconds.append(time.monotonic()-st);mon.sample()
    outputs[device]=np.stack(out);times[device]=seconds
   original=c.load(c.B/name/'features.npz');np.testing.assert_array_equal(outputs['mps'],original['X'][[p['IDs'].index(i) for i in p['smoke_IDs']]]);absdiff=float(abs(outputs['cpu']-outputs['mps']).max());rel=float(np.linalg.norm(outputs['cpu']-outputs['mps'])/max(np.linalg.norm(outputs['cpu']),1e-12));assert absdiff<.001 and rel<.001,(absdiff,rel)
   assert initial==[c.statehash(x) for x in [prefix,tail]]
   c.arrays(R/f'{name}-smoke.npz',ids=p['smoke_IDs'],**outputs)
   records[name]={'finite':True,'CPU_MPS_maxabs':absdiff,'CPU_MPS_relative_L2':rel,'seconds':times,'weights_unchanged':True,'projected400_MPS_seconds':400*float(np.median(times['mps']))};assert records[name]['projected400_MPS_seconds']<600
   c.emit({name:records[name]});del prefix,tail;gc.collect();torch.mps.empty_cache()
 c.write(R/'smoke.json',{'passed':True,'batch':1,'dtype':'float32','IDs':p['smoke_IDs'],'models':records,'peak_conservative_RSS_plus_MPS':max(z['RSS_plus_MPS'] for z in mon.samples),'swap_growth':max(z['swap'] for z in mon.samples)-mon.initialswap,'network_attempts':len(c.network_attempts)})

def extract(name):
 p=c.check();assert c.read(R/'smoke.json')['passed'];out=R/name;out.mkdir();rows=[];hashes={};st=time.monotonic()
 with c.Monitor(name+'-extract',600) as mon:
  prefix,tail=c.model(name);initial=[c.statehash(x) for x in [prefix,tail]];prefix.to('mps');tail.to('mps');cache=None
  for j,i in enumerate(p['IDs']):
   with torch.inference_mode():q=prefix(c.image_input(i).to('mps'));a=tail(q).cpu().numpy()[0];q=q.cpu().numpy()[0]
   if cache is None:cache=np.lib.format.open_memmap(out/'prefix.npy',mode='w+',dtype='float32',shape=(400,*q.shape))
   cache[j]=q;rows.append(a);hashes[i]=c.sha(P/'data/train'/f'{i}.png')
   if (j+1)%80==0:c.emit({'model':name,'rows':j+1,'seconds':time.monotonic()-st});mon.sample()
  cache.flush();shape=list(cache.shape);del cache;X=np.stack(rows);assert np.isfinite(X).all();assert initial==[c.statehash(x) for x in [prefix,tail]]
  smoke=c.load(R/f'{name}-smoke.npz')
  for j,i in enumerate(smoke['ids']):np.testing.assert_array_equal(X[p['IDs'].index(i)],smoke['mps'][j])
  old=c.read(P/'artifacts/dinob-062/extraction.json');assert hashes==old['image_sha256']
  original=c.load(c.B/name/'features.npz');np.testing.assert_array_equal(original['ids'],p['IDs']);np.testing.assert_array_equal(X,original['X']);c.arrays(out/'features.npz',ids=p['IDs'],X=X);del prefix,tail;gc.collect();torch.mps.empty_cache()
 rec={'passed':True,'shape':list(X.shape),'prefix_shape':shape,'image_hashes':hashes,'feature_sha256':c.sha(out/'features.npz'),'prefix_sha256':c.sha(out/'prefix.npy'),'seconds':time.monotonic()-st,'weights_unchanged':True,'smoke_exact':True,'image_hashes_match_prior':True,'all400_frozen_features_exact185':True};c.write(out/'extraction.json',rec);c.emit({k:v for k,v in rec.items() if k!='image_hashes'});c.check()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','resnet50','efficientnet_b0']);a=ap.parse_args();smoke() if a.mode=='smoke' else extract(a.mode)
