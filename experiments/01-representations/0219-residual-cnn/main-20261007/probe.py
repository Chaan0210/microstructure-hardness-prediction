from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,time,json,resource,subprocess
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
O=Path(__file__).resolve().parent;S=O.parent;R=_RepoPath(__file__).resolve().parents[4];sys.path.insert(0,str(S))
import numpy as np,torch
from threadpoolctl import threadpool_limits
from data import load_development,graph_for
from backend import TinyCNN,cnn_normalization,resource_check
from contracts import sha,lock_manifest,require
threadpool_limits(2);torch.set_num_threads(2);require(torch.backends.mps.is_available(),'MPS unavailable')
f,labels,contract=load_development();E=np.array(graph_for(0).cells[0]['deep'][0]['E']);y=labels.take(E);X=f.take('images',E);mu,sd,ym,ys=cnn_normalization(X,y);X=(X-mu)/sd;y=(y-ym)/ys;torch.manual_seed(20263999);rng=np.random.default_rng(20263999);m=TinyCNN().to('mps');opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01);times=[];losses=[];memory=[];start=time.perf_counter()
while len(times)<8:
 order=rng.permutation(len(E))
 for first in range(0,len(E),32):
  if len(times)==8:break
  ii=order[first:first+32];x=np.rot90(X[ii],int(rng.integers(4)),axes=(2,3))
  if rng.random()<.5:x=x[:,:,:,::-1]
  torch.mps.synchronize();t=time.perf_counter();xx=torch.from_numpy(np.ascontiguousarray(x)).to('mps');yy=torch.tensor(y[ii],dtype=torch.float32,device='mps');opt.zero_grad(set_to_none=True);loss=(m(xx)-yy).square().mean();require(bool(torch.isfinite(loss)),'Probe loss');loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step();torch.mps.synchronize();times.append(time.perf_counter()-t);losses.append(float(loss.detach().cpu()));memory.append(resource_check('mps'))
steady=float(np.median(times[2:]));estimate=steady*44080;result={'passed':True,'updates':8,'training_rows':len(E),'seed':20263999,'seconds':time.perf_counter()-start,'update_seconds':times,'steady_median_seconds':steady,'CNN_main_plus_verification_estimate_seconds':estimate,'whole_study_estimate_seconds':[estimate+180,estimate+600],'loss_standardized':losses,'memory':memory,'swap':subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True).strip(),'protocol_sha256':sha(_repo_path(S, 'protocol.json')),'probe_weights_discarded':True};del m,opt;torch.mps.empty_cache();require(max(z['RSS_peak_bytes']+z['MPS_driver_bytes'] for z in memory)<6e9,'Probe memory exceeded');result['within_25min_forecast']=estimate+600<=1500;lock_manifest(_repo_path(O, 'probe.json'),result);print(json.dumps(result),flush=True)
