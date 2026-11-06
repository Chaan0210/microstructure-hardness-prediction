"""Descriptive geometry/noise sensitivity; never a hardness selection gate."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
import amplitude_cnn_106 as s
import dinob_062 as b

def distances(real,fixture):
 sd=real.std(0);active=sd>1e-8;Z=(fixture[:,active]-real[:,active].mean(0))/sd[active];D=np.sqrt(np.mean((Z[:,None]-Z[None,:])**2,axis=2));same=[D[i,i+1] for i in (0,2,4)];different=[D[i,j] for noise in (0,1) for i,j in ((noise,2+noise),(noise,4+noise),(2+noise,4+noise))]
 return {'dimensions':real.shape[1],'active_dimensions':int(active.sum()),'distance_matrix':D.tolist(),'same_geometry_changed_noise':list(map(float,same)),'different_geometry_matched_noise':list(map(float,different)),'median_same_noise_change':float(np.median(same)),'median_different_geometry':float(np.median(different)),'geometry_to_noise_distance_ratio':float(np.median(different)/np.median(same))}
def main():
 start=time.perf_counter();s.check();torch.set_num_threads(2);d=s.inputs('original');tr=np.flatnonzero(d['fid']!=0);paths=[_repo_path(ROOT, f'artifacts/nlm-measurement-098/verified-fixtures/case-{k}.npz') for k in range(6)];raw=[]
 for p in paths:
  with np.load(_repo_io(p),allow_pickle=False) as z:raw.append(z['raw'].copy())
 fx={'X':np.stack(raw)[:,None]};result={};saved={}
 for arm in s.ARMS:
  path=s.OUT/'original'/f'{arm}-fold0/model.pt';a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);m=s.Net(arm);m.load_state_dict(a['state']);m=m.to('mps');real=s.predict(m,d,tr,a['normalization'],descriptors=True);f=s.predict(m,fx,np.arange(6),a['normalization'],descriptors=True);result[arm]=distances(real,f);saved[arm+'_real']=real;saved[arm+'_fixture']=f;del m;torch.mps.empty_cache()
 path=_repo_path(ROOT, 'artifacts/dinob-062/development_features.npz')
 with np.load(_repo_io(path),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];real=z['X'][tr].astype(float)
 m=b.model().to('mps');arr=[]
 for x in raw:
  z=np.array(Image.fromarray(x).resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/np.float32(255);a=(np.repeat(z[None],3,axis=0)-b.ds.base.MEAN)/b.ds.base.STD
  with torch.inference_mode():arr.append(b.feature(m,torch.from_numpy(a[None]).to('mps')).cpu().numpy()[0])
 f=np.stack(arr).astype(float);result['frozen_DINO_B']=distances(real,f);saved['DINO_B_real']=real;saved['DINO_B_fixture']=f
 with (s.OUT/'representation-descriptors.npz').open('xb') as file:np.savez_compressed(_repo_io(file),train_ids=np.array(d['ids'])[tr],**saved)
 # Distance matrix independently recomputed by explicit per-pair standardized differences.
 for name in ('groupnorm','affine','frozen_DINO_B'):
  prefix='DINO_B' if name=='frozen_DINO_B' else name;r=saved[prefix+'_real'];f=saved[prefix+'_fixture'];sd=np.std(r,axis=0);ix=sd>1e-8
  for i in range(6):
   for j in range(6):assert abs(float(np.linalg.norm((f[i,ix]-f[j,ix])/sd[ix])/np.sqrt(ix.sum()))-result[name]['distance_matrix'][i][j])<1e-10
 report={'descriptive_only':True,'results':result,'fixture_order':['coarse-low-noise','coarse-high-noise','fine-low-noise','fine-high-noise','elongated-low-noise','elongated-high-noise'],'normalization':'Feature standard deviations fit original fold0 train320 real descriptors only; constant columns excluded at fixed1e-8. RMS standardized Euclidean distance, no fitted geometry predictor.','limitations':['Only six deliberately simple known finite-window geometries.','Different layouts also differ in phase/intensity realizations; distance is not isolated size/topology sensitivity.','Larger ratio does not prove geometry recovery, real-noise robustness, or predict hardness gain.','Global moments are many-to-one; CNN stride16 loses exact fine arrangement. DINO tokens include position/context, but final pooling cannot reconstruct full topology.'],'no_synthetic_training':True,'no_hardness_labels_used_in_distance_analysis':True,'fixtures_sha256':{str(p.relative_to(ROOT)):s.sha(p) for p in paths},'source_sha256':s.sha(Path(__file__)),'descriptors_sha256':s.sha(s.OUT/'representation-descriptors.npz'),'distance_independent_recompute_passed':True,'seconds':time.perf_counter()-start,'holdout_test_used':False};s.write(s.OUT/'representation-review.json',report);s.emit(report);s.check()
if __name__=='__main__':main()
