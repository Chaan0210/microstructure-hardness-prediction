"""Development-only final L token adjacency, same018 statistics and fixed QR rule."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import gc,time,sys,os
import numpy as np,torch
import runtime as f
R=f.R;P=f.P

def projection(width):return np.linalg.qr(np.random.default_rng(20271201).normal(size=(width,32)),mode='reduced')[0]
def descriptor(tokens,Q):
 z=(tokens.astype(float)@Q).reshape(36,36,32);h=np.diff(z,axis=0).reshape(-1,32);v=np.diff(z,axis=1).reshape(-1,32)
 return ((h.T@h+v.T@v)/(2*(len(h)+len(v))))[np.triu_indices(32)]
def independent(tokens,Q):
 t=tokens.reshape(36,36,-1).astype(float);edges=np.array([t[a,b]-t[c,d] for a in range(36) for b in range(36) for c,d in [(a+1,b),(a,b+1)] if c<36 and d<36]);assert len(edges)==2520;e=edges@Q
 return np.einsum('ni,nj->ij',e,e)[np.triu_indices(32)]/(2*2520)
def forward(m,x):
 with torch.inference_mode():
  p,c=m.get_intermediate_layers(x,n=[23],return_class_token=True,norm=True)[0];assert p.shape==(1,1296,1024)
  glob=torch.cat([c,p.mean(1),p.std(1,correction=0)],1).cpu().numpy()[0];tok=p.cpu().numpy()[0]
 return tok,glob

def main():
 plan=f.check();allowed={P/'data/train'/f'{i}.png' for i in plan['IDs']}
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   pp=f.Path(os.fsdecode(args[0])).absolute()
   if pp.is_relative_to(P/'data') and pp not in allowed:raise RuntimeError('Holdout/test/rawCSV forbidden: '+str(pp))
 sys.addaudithook(guard);start=time.perf_counter();Q=projection(1024);np.testing.assert_allclose(Q.T@Q,np.eye(32),atol=1e-14,rtol=0);checks={};rows=[];hashes={}
 with f.Monitor('spatial-extraction',plan['resources']):
  # Exact historical B projection and independent edge math on three cached development examples.
  meta=f.read(P/'artifacts/patch-attention-092/cache.json');path=P/'artifacts/patch-attention-092/patch_tokens.npy';assert f.sha(path)==meta['token_sha256'];tok=np.load(_repo_io(path),mmap_mode='r');Bq=projection(768)
  with np.load(_repo_io(P/'artifacts/spatial-joint-140/features.npz'),allow_pickle=False) as z:
   assert z['ids'].tolist()==plan['IDs'];np.testing.assert_array_equal(Bq,z['projection']);err=0.
   for i in [0,199,399]:
    a=descriptor(tok[i],Bq);np.testing.assert_array_equal(a,z['adjacent'][i]);err=max(err,float(abs(a-independent(tok[i],Bq)).max()))
   assert err<1e-12;checks['B_3cached_exact']=True;checks['B_independent_edge_max_error']=err
  del tok;gc.collect();m=f.model();initial=f.statehash(m);smokes={};drift=0.;edgeerr=0.;shuffle_effect=[]
  for device in ['cpu','mps']:
   m=m.to(device);smokes[device]=[]
   for i in plan['smoke_IDs']:
    token,glob=forward(m,torch.from_numpy(f.image_input(i)[None]).to(device));desc=descriptor(token,Q);smokes[device].append(desc);edgeerr=max(edgeerr,float(abs(desc-independent(token,Q)).max()))
    if device=='mps':
     for grid in [np.rot90(token.reshape(36,36,1024)),token.reshape(36,36,1024)[:,::-1]]:np.testing.assert_allclose(descriptor(grid.reshape(1296,1024),Q),desc,atol=1e-12,rtol=0)
     perm=np.random.default_rng(202710179).permutation(1296);shuffled=token[perm];np.testing.assert_allclose(shuffled.astype(float).mean(0),token.astype(float).mean(0),atol=1e-14,rtol=0);np.testing.assert_allclose(shuffled.astype(float).std(0),token.astype(float).std(0),atol=1e-12,rtol=0);shuffle_effect.append(float(np.linalg.norm(descriptor(shuffled,Q)-desc)))
   smokes[device]=np.stack(smokes[device])
  drift=float(abs(smokes['cpu']-smokes['mps']).max());assert drift<plan['tolerances']['CPU_MPS_spatial_feature'];assert edgeerr<1e-12 and min(shuffle_effect)>1e-5
  checks.update(CPU_MPS_spatial_max_error=drift,L_independent_edge_max_error=edgeerr,token_grid_D4_invariant=True,whole_image_DINO_D4_invariance_claimed=False,shuffle_spatial_L2_changes=shuffle_effect,shuffle_global_mean_std_invariant=True)
  with np.load(_repo_io(P/'experiments/dinol-177/features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==plan['IDs'];old=z['X'][:,6144:].copy()
  for j,i in enumerate(plan['IDs']):
   token,glob=forward(m,torch.from_numpy(f.image_input(i)[None]).to('mps'));np.testing.assert_array_equal(glob,old[j]);rows.append(descriptor(token,Q));hashes[i]=f.sha(P/'data/train'/f'{i}.png')
   if (j+1)%80==0:f.emit({'completed':j+1,'seconds':time.perf_counter()-start})
  X=np.stack(rows);assert X.shape==(400,528) and np.isfinite(X).all();assert f.statehash(m)==initial;assert hashes==f.read(P/'experiments/dinol-177/extraction.json')['image_sha256']
  for j,i in enumerate(plan['smoke_IDs']):np.testing.assert_array_equal(X[plan['IDs'].index(i)],smokes['mps'][j])
  del m;torch.mps.empty_cache()
  with (R/'features.npz').open('xb') as ff:np.savez_compressed(_repo_io(ff),ids=plan['IDs'],X=X,projection=Q)
 rec={'passed':True,'rows':400,'shape':[400,528],'feature_sha256':f.sha(R/'features.npz'),'projection_sha256':f.hashlib.sha256(Q.tobytes()).hexdigest(),'validation':checks,'global_L_final_stats_exact400':True,'raw400_hashes_exact':True,'backbone_state_unchanged':True,'smoke_repeat_exact':True,'no_holdout_test_labels':True,'seconds':time.perf_counter()-start};f.write(R/'feature-validation.json',rec);f.emit(rec)
if __name__=='__main__':
 try:main()
 except Exception as e:f.write(R/'extraction-failure.json',{'error':repr(e)});raise
