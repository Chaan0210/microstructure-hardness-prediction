from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,np,torch
import argparse,time,gc

def descriptor(tail,q):
 h=tail[0](q);mean=tail[1:](h).flatten(1);std=h.std(dim=(2,3),correction=0)
 assert h.shape[-2:]==(8,8) and torch.isfinite(h).all() and torch.isfinite(mean).all() and torch.isfinite(std).all()
 return mean,std

def run(name):
 c.check();root=R/name;root.mkdir();start=time.monotonic();source=P/c.plan['cache_roots'][name];cached=c.load(source/'features.npz');assert cached['ids'].tolist()==c.plan['IDs'];cache=np.load(_repo_io(source/'prefix.npy'),mmap_mode='r');assert len(cache)==400
 with c.Monitor(name+'-features',600) as mon:
  prefix,tail=c.model(name);hashes=[c.statehash(m) for m in [prefix,tail]];smoke={}
  for device in ['cpu','mps']:
   prefix.to(device);tail.to(device);rows=[]
   for i in c.plan['smoke_IDs']:
    with torch.inference_mode():a,b=descriptor(tail,prefix(c.image_input(i).to(device)))
    rows.append(torch.cat([a,b],1).cpu().numpy()[0]);mon.sample()
   smoke[device]=np.stack(rows)
  diff=float(abs(smoke['cpu']-smoke['mps']).max());rel=float(np.linalg.norm(smoke['cpu']-smoke['mps'])/max(np.linalg.norm(smoke['cpu']),1e-12));assert diff<.001 and rel<.001
  means=[];stds=[]
  for j in range(400):
   with torch.inference_mode():a,b=descriptor(tail,torch.from_numpy(np.array(cache[j:j+1])).to('mps'))
   means.append(a.cpu().numpy()[0]);stds.append(b.cpu().numpy()[0])
   if (j+1)%100==0:c.emit({'extracted':name,'rows':j+1,'seconds':time.monotonic()-start})
  mean=np.stack(means);std=np.stack(stds);np.testing.assert_array_equal(mean,cached['X']);assert np.isfinite(std).all() and (std>=0).all();assert hashes==[c.statehash(m) for m in [prefix,tail]]
  for j,i in enumerate(c.plan['smoke_IDs']):np.testing.assert_array_equal(np.r_[mean[c.plan['IDs'].index(i)],std[c.plan['IDs'].index(i)]],smoke['mps'][j])
  c.arrays(root/'features.npz',ids=cached['ids'],mean=mean,std=std);c.arrays(root/'smoke.npz',ids=c.plan['smoke_IDs'],**smoke)
  rec={'passed':True,'shape_mean':list(mean.shape),'shape_std':list(std.shape),'spatial_size':[8,8],'CPU_MPS_descriptor_maxabs':diff,'CPU_MPS_relative_L2':rel,'existing_mean_cache_exact':True,'fixed3_smoke_vs_cached_prefix_exact':True,'weights_frozen_and_unchanged':True,'features_SHA256':c.sha(root/'features.npz'),'source_prefix_SHA256':c.sha(source/'prefix.npy'),'seconds':time.monotonic()-start,'no_new_weights':True};c.write(root/'extraction.json',rec);c.emit(rec);del prefix,tail;gc.collect();torch.mps.empty_cache()
 c.check()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',choices=c.plan['models']);run(p.parse_args().model)
