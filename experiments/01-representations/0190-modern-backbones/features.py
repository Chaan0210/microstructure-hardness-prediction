from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,np,torch
from torchvision.transforms import functional as TF
from PIL import Image
import argparse,time,gc

def preprocessing_check(ident):
 x=c.image_input(ident)[0]
 with Image.open(_repo_io(P/'data/train'/f'{ident}.png')) as im:
  raw=np.asarray(im).copy();rgb=TF.pil_to_tensor(im.convert('RGB')).float()/255
 p=c.read(R/'provenance.json');mean=p['weights']['convnext_tiny']['mean'];std=p['weights']['convnext_tiny']['std'];assert mean==p['weights']['swin_v2_t']['mean'] and std==p['weights']['swin_v2_t']['std'];official=TF.normalize(rgb,mean,std);error=float(abs(x-official).max());assert error<=1e-6 and x.shape==(3,256,256);inv=(x*torch.tensor(std)[:,None,None]+torch.tensor(mean)[:,None,None]);np.testing.assert_array_equal((inv[0].numpy()*255).round().astype('uint8'),raw);return error

def smoke(name):
 c.check();records={};start=time.monotonic()
 with c.Monitor(name+'-smoke',300) as mon:
  prefix,tail=c.model(name);initial=[c.statehash(q) for q in [prefix,tail]];outputs={};times={};shapes=[];preprocess=[]
  for device in ['cpu','mps']:
   prefix.to(device);tail.to(device);rows=[];dur=[]
   for ident in c.plan['smoke_IDs']:
    preprocess.append(preprocessing_check(ident));st=time.monotonic()
    with torch.inference_mode():q=prefix(c.image_input(ident).to(device));a=tail(q);again=tail(q);assert torch.equal(a,again);arr=a.cpu().numpy()[0]
    assert arr.shape==(768,) and np.isfinite(arr).all();rows.append(arr);dur.append(time.monotonic()-st);shapes.append(list(q.shape));mon.sample()
   outputs[device]=np.stack(rows);times[device]=dur
  absolute=float(abs(outputs['cpu']-outputs['mps']).max());relative=float(np.linalg.norm(outputs['cpu']-outputs['mps'])/max(np.linalg.norm(outputs['cpu']),1e-12));assert absolute<.001 and relative<.001
  # Hardware-only backward test: no labels or optimizer step, weights remain unchanged.
  tail.requires_grad_(True);tail.eval();x=c.image_input(c.plan['smoke_IDs'][0]).to('mps')
  with torch.no_grad():q=prefix(x)
  a=tail(q);loss=(a*torch.linspace(-1,1,768,device='mps')).mean();loss.backward();grads=[p.grad for p in tail.parameters() if p.requires_grad];assert all(g is not None and torch.isfinite(g).all() for g in grads);gradnorm=float(torch.stack([g.norm()**2 for g in grads]).sum().sqrt());assert np.isfinite(gradnorm) and gradnorm>0;tail.zero_grad(set_to_none=True);tail.requires_grad_(False);assert initial==[c.statehash(q) for q in [prefix,tail]];mon.sample()
  c.arrays(R/f'{name}-smoke.npz',ids=c.plan['smoke_IDs'],**outputs);record={'passed':True,'model':name,'batch':1,'dtype':'float32','CPU_MPS_maxabs':absolute,'CPU_MPS_relative_L2':relative,'MPS_repeat_exact':True,'weights_unchanged':True,'no_step_MPS_backward_finite':True,'backward_gradient_norm':gradnorm,'prefix_shapes':shapes,'all_pixel_preprocessing_maxabs':max(preprocess),'full_original_pixels_recovered_exact':True,'seconds_per_image':times,'projected400_MPS_seconds':400*float(np.median(times['mps'])),'trainable_tail_parameters':sum(p.numel() for p in tail.parameters()),'network_attempts':len(c.network_attempts),'seconds':time.monotonic()-start};assert record['projected400_MPS_seconds']<600 and not c.network_attempts;c.write(R/f'{name}-smoke.json',record);c.emit(record);del prefix,tail,q,x,a,grads;gc.collect();torch.mps.empty_cache()

def extract(name):
 c.check();assert c.read(R/f'{name}-smoke.json')['passed'];root=R/name;root.mkdir();rows=[];hashes={};start=time.monotonic()
 with c.Monitor(name+'-extract',600) as mon:
  prefix,tail=c.model(name);initial=[c.statehash(q) for q in [prefix,tail]];prefix.to('mps');tail.to('mps');cache=None
  for j,ident in enumerate(c.plan['IDs']):
   with torch.inference_mode():q=prefix(c.image_input(ident).to('mps'));a=tail(q).cpu().numpy()[0];q=q.cpu().numpy()[0]
   if cache is None:cache=np.lib.format.open_memmap(root/'prefix.npy',mode='w+',dtype='float32',shape=(400,*q.shape))
   assert np.isfinite(q).all() and np.isfinite(a).all();cache[j]=q;rows.append(a);hashes[ident]=c.sha(P/'data/train'/f'{ident}.png')
   if (j+1)%80==0:c.emit({'model':name,'rows':j+1,'seconds':time.monotonic()-start});mon.sample()
  cache.flush();shape=list(cache.shape);del cache;X=np.stack(rows);assert X.shape==(400,768);sm=c.load(R/f'{name}-smoke.npz')
  for j,ident in enumerate(sm['ids']):np.testing.assert_array_equal(X[c.plan['IDs'].index(ident)],sm['mps'][j])
  assert hashes==c.read(P/'artifacts/dinob-062/extraction.json')['image_sha256'];assert initial==[c.statehash(q) for q in [prefix,tail]];c.arrays(root/'features.npz',ids=c.plan['IDs'],X=X);del prefix,tail;gc.collect();torch.mps.empty_cache()
 record={'passed':True,'feature_shape':list(X.shape),'prefix_shape':shape,'feature_SHA256':c.sha(root/'features.npz'),'prefix_SHA256':c.sha(root/'prefix.npy'),'image_hashes':hashes,'input_hashes_match_prior':True,'smoke_features_exact':True,'weights_unchanged':True,'seconds':time.monotonic()-start};c.write(root/'extraction.json',record);c.emit({k:v for k,v in record.items() if k!='image_hashes'});c.check()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','extract']);p.add_argument('model',choices=['convnext_tiny','swin_v2_t']);a=p.parse_args();smoke(a.model) if a.mode=='smoke' else extract(a.model)
