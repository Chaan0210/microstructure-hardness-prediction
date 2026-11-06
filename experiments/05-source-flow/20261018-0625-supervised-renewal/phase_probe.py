from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,time,resource
import numpy as np
import torch
from torch import nn
import phase_joint_v2 as p

H=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.35)
d=dict(np.load(_repo_path(H, 'data.npz')));px=np.load(_repo_path(H, 'train-pixels.npy'),mmap_mode='r');tr=d['dev'][d['fold']!=0]
aux=p.prepare(d,px,tr,_repo_path(H, 'runs/phase-joint-f0-s81'));rng=np.random.default_rng(2026101881);torch.manual_seed(2026101881)
net=p.UNet(32).to('mps');opt=torch.optim.AdamW(net.parameters(),lr=.001);times=[]
for step in range(10):
    t=time.monotonic();loss=p.aux_loss(net,p.aux_batch(aux,px,rng,batch=4),aux);opt.zero_grad(set_to_none=True);loss.backward();opt.step();torch.mps.synchronize();times.append(time.monotonic()-t)
pre_memory=dict(peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,MPS_driver=torch.mps.driver_allocated_memory())
model=p.Joint(net,aux,np.zeros(19),np.ones(19),np.ones(19)*.01,0.);opt=torch.optim.AdamW(model.parameters(),lr=.0001);jt=[]
for step in range(5):
    t=time.monotonic();model.train()
    for layer in model.net.modules():
        if isinstance(layer,nn.BatchNorm2d):layer.eval()
    ix=tr[step*4:step*4+4];x=p.tensor(np.array([p.norm(px[j,0]) for j in ix])[:,None]);y=p.tensor((d['y'][ix]-d['y'][tr].mean())/d['y'][tr].std())
    loss=(model(x,torch.zeros(4,0,device='mps'))-y).square().mean()+.25*p.aux_loss(net,p.aux_batch(aux,px,rng,batch=2),aux)
    opt.zero_grad(set_to_none=True);loss.backward();opt.step();torch.mps.synchronize();jt.append(time.monotonic()-t)
fine_memory=dict(peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,MPS_driver=torch.mps.driver_allocated_memory())
r=dict(pretrain_batch=4,hardness_batch=4,auxiliary_batch=2,pretrain_seconds_per_step=float(np.median(times[2:])),fine_seconds_per_step=float(np.median(jt[1:])),pretrain_memory=pre_memory,fine_memory=fine_memory,
       projected_seconds=float(np.median(times[2:])*4000+np.median(jt[1:])*80*80),validation_images_read=False,benchmark_updates_discarded=True)
r['passed']=max(sum(pre_memory.values()),sum(fine_memory.values()))<6*2**30
(_repo_path(H, 'phase-resource-probe-b4.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
assert r['passed']
