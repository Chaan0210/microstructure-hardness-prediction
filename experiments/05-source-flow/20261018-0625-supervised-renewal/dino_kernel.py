"""Train DINOv2's last block through a newly regularized analytic prediction head.

Frozen official prefix is independent of this competition. Only outer-training
images enter the learned tail, feature scaling, regularization selection or head.
Analytic LOO is a training proxy, not independent validation of the learned tail.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'));os.environ['XFORMERS_DISABLED']='1';os.environ['HF_HUB_OFFLINE']='1'
import argparse,copy,hashlib,json,math,resource,time
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import deep_kernel as dk
VENDOR=_repo_path(ROOT, 'vendor/dinov2-base-062');PROVENANCE=json.loads((_repo_path(VENDOR, 'provenance.json')).read_text())
sys.path.insert(0,str(_repo_path(ROOT, PROVENANCE['official_source_dir'])))
from dinov2.hub.backbones import dinov2_vitb14

class Tail(nn.Module):
    def __init__(self):
        super().__init__();m=dinov2_vitb14(pretrained=False);m.load_state_dict(torch.load(_repo_path(VENDOR, 'dinov2_vitb14_pretrain.pth'),map_location='cpu',weights_only=True),strict=True)
        self.block=m.blocks[-1];self.norm=m.norm
    def forward(self,z,early):
        z=self.norm(self.block(z));p=z[:,1:]
        return torch.cat([early,z[:,0],p.mean(1),p.std(1,correction=0)],1)

def load():
    d=np.load(_repo_path(HERE, 'data.npz'));record=json.loads((_repo_path(ROOT, 'artifacts/dinob-adapter-080/prefix.json')).read_text());f=np.load(_repo_path(ROOT, 'artifacts/dinob-062/development_features.npz'))
    assert np.array_equal(f['ids'],d['ids'][d['dev']]);assert f['ids'].tolist()==record['ids']
    tokens=np.load(_repo_path(ROOT, 'artifacts/dinob-adapter-080/prefix.npy'),mmap_mode='r');assert tokens.shape==(400,1297,768)
    return d,tokens,f['X'].astype(np.float32),record

def batches(model,tokens,features,ix,device,batch):
    for first in range(0,len(ix),batch):
        jj=ix[first:first+batch]
        yield model(torch.tensor(np.array(tokens[jj]),device=device),torch.tensor(features[jj,:4608],device=device))

@torch.no_grad()
def extract(model,tokens,features,ix,device,batch):return np.concatenate([x.cpu().numpy() for x in batches(model,tokens,features,ix,device,batch)])

def check(a):
    d,tokens,features,record=load();model=Tail().eval().to(a.device);ix=np.flatnonzero(d['fold']!=0)[:4]
    y=torch.tensor(d['y'][d['dev'][ix]],dtype=torch.float64);y=(y-y.mean())/y.std(correction=0)
    torch.manual_seed(a.seed);start=time.monotonic();f=next(batches(model,tokens,features,ix,a.device,4));initial=f.detach().cpu().numpy();diff=float(abs(initial-features[ix]).max());assert diff<.002
    loss=dk.training_proxy(f.cpu().double(),y,.1);loss.backward();g={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters()};model.zero_grad(set_to_none=True)
    cache=torch.tensor(initial,dtype=torch.float64,requires_grad=True);other=dk.training_proxy(cache,y,.1);other.backward()
    for j,z in enumerate(batches(model,tokens,features,ix,a.device,2)):z.backward(cache.grad[j*2:j*2+2].float().to(a.device))
    gd=max(float((g[k]-p.grad.cpu()).abs().max()) for k,p in model.named_parameters());assert gd<.002
    memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+(torch.mps.driver_allocated_memory() if a.device=='mps' else 0)
    r=dict(frozen_descriptor_max_error=diff,gradient_cache_max_error=gd,loss_difference=abs(float(loss.detach()-other.detach())),seconds=time.monotonic()-start,memory_bytes=memory,optimizer_updates=0,training_ids=d['ids'][d['dev'][ix]].tolist())
    dk.direct.save(_repo_path(HERE, 'dino-kernel-check.json'),r);print(json.dumps(r),flush=True)

def train(a):
    d,tokens,features,prefix=load();tr=np.flatnonzero(d['fold']!=a.fold);q=np.flatnonzero(d['fold']==a.fold);yall=d['y'][d['dev']]
    torch.manual_seed(a.seed)
    if a.device=='mps':torch.mps.manual_seed(a.seed);torch.mps.set_per_process_memory_fraction(.35)
    model=Tail().eval().to(a.device);prior=[p.detach().clone() for p in model.parameters()]
    out=_repo_path(HERE, 'runs', a.tag);out.mkdir(exist_ok=True);assert not (out/'done.json').exists()
    fresh=extract(model,tokens,features,tr[:2],a.device,a.batch);assert abs(fresh-features[tr[:2]]).max()<.002
    alpha,choice=dk.select_alpha(features[tr],yall[tr],2026101913+a.fold);head=dk.fit_head(features[tr],yall[tr],alpha);initial=dk.predict_head(head,features[q])
    ym=float(yall[tr].mean());ys=float(yall[tr].std());y=torch.tensor((yall[tr]-ym)/ys,dtype=torch.float64)
    recipe=vars(a)|dict(train_ids=d['ids'][d['dev'][tr]].tolist(),query_ids=d['ids'][d['dev'][q]].tolist(),alpha=alpha,alpha_selection=choice,
        pretrained=PROVENANCE,prefix_sha256=prefix['sha256'],prefix_provenance='Existing frozen official prefix with all400 descriptor equality in prefix.json; no competition fitting.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kernel_source_sha256=hashlib.sha256((_repo_path(HERE, 'deep_kernel.py')).read_bytes()).hexdigest(),
        hypothesis='Earlier tail training retained nearly interpolating frozen MLP heads and a teacher prediction anchor. This trains lastblock+norm through a newly fitted regularized analytic head, without those heads/anchors.',
        objective='Analytic ridge LOO training proxy, not unbiased validation of learned features; full outer-train gradient cached in microbatches.',
        preprocessing='Original full504 view; frozen normalized block6/9 descriptors plus trainable block12 CLS/mean/std. Train-only mean/RMS-floor scales.',
        averaging='Equal parameter mean of final10 steps, fixed before results; final analytic head refit on outer train. No best-outer-checkpoint selection.')
    dk.direct.save(_repo_io(out/'recipe.json'),recipe);np.savez(_repo_io(out/'initial.npz'),query=d['dev'][q],prediction=initial);print('INITIAL',dk.direct.rmse(initial,yall[q]),'alpha',alpha,flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0);start=time.monotonic();curve=[];first=0;elapsed=0;average=None;navg=0
    if (out/'resume.pt').exists():
        r=torch.load(_repo_io(out/'resume.pt'),map_location='cpu',weights_only=False);assert r['recipe']==recipe;model.load_state_dict(r['state']);opt.load_state_dict(r['optimizer']);first=r['step'];curve=r['curve'];elapsed=r['elapsed'];average=r['average'];navg=r['navg']
        for state in opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value) and key!='step':state[key]=value.to(a.device)
    for step in range(first,a.steps):
        model.eval();F=extract(model,tokens,features,tr,a.device,a.batch);leaf=torch.tensor(F,dtype=torch.float64,requires_grad=True);loss=dk.training_proxy(leaf,y,alpha);assert torch.isfinite(loss);loss.backward();grad=leaf.grad.float()
        opt.zero_grad(set_to_none=True)
        for j,z in enumerate(batches(model,tokens,features,tr,a.device,a.batch)):z.backward(grad[j*a.batch:j*a.batch+len(z)].to(a.device))
        anchor=a.anchor*sum((p-old).square().sum() for p,old in zip(model.parameters(),prior));anchor.backward();nn.utils.clip_grad_norm_(model.parameters(),5.)
        for g in opt.param_groups:g['lr']=a.lr*(.05+.95*.5*(1+math.cos(math.pi*step/a.steps)))
        opt.step()
        if a.device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
        if step>=a.steps-10:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};navg+=1
            if average is None:average=state
            else:
                for k,v in state.items():average[k].add_((v-average[k])/navg)
        row=dict(step=step+1,training_LOO_proxy_rmse_before_update=float(loss.detach().sqrt())*ys,seconds=elapsed+time.monotonic()-start)
        if step==0 or (step+1)%10==0 or step+1==a.steps:
            T=extract(model,tokens,features,tr,a.device,a.batch);Q=extract(model,tokens,features,q,a.device,a.batch);h=dk.fit_head(T,yall[tr],alpha);pred=dk.predict_head(h,Q)
            row.update(validation_rmse=dk.direct.rmse(pred,yall[q]),training_fitted_rmse=dk.direct.rmse(dk.predict_head(h,T),yall[tr]));np.savez(_repo_io(out/f'step{step+1:03d}.npz'),query=d['dev'][q],prediction=pred)
            r=dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),recipe=recipe,step=step+1,curve=curve+[row],elapsed=row['seconds'],average=average,navg=navg)
            torch.save(r,_repo_io(out/'resume-new.pt'));(out/'resume-new.pt').replace(out/'resume.pt');print(json.dumps(row),flush=True)
        curve.append(row);dk.direct.save(_repo_io(out/'curve.json'),curve)
        memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+(torch.mps.driver_allocated_memory() if a.device=='mps' else 0);assert memory<6*2**30
    model.load_state_dict(average);T=extract(model,tokens,features,tr,a.device,a.batch);Q=extract(model,tokens,features,q,a.device,a.batch);head=dk.fit_head(T,yall[tr],alpha);pred=dk.predict_head(head,Q)
    torch.save(dict(state=average,recipe=recipe),_repo_io(out/'model.pt'));np.savez(_repo_io(out/'head.npz'),**head);np.savez(_repo_io(out/'predictions.npz'),train=d['dev'][tr],query=d['dev'][q],prediction=pred,train_features=T,query_features=Q)
    ref=np.load(_repo_path(ROOT, 'experiments/rmse-research-20261015/inputs.npz'));base=d['base083'][q];p21=ref['C73_cnn'][0,q];eligible=~d['failed'][q]
    r=dict(rmse=dk.direct.rmse(pred,yall[q]),frozen_rmse=dk.direct.rmse(initial,yall[q]),base083_rmse=dk.direct.rmse(base,yall[q]),quarter_blend_rmse=dk.direct.rmse(base+.25*eligible*(pred-base),yall[q]),replace_image_branch_rmse=dk.direct.rmse(base+.16*eligible*(pred-p21),yall[q]),seconds=elapsed+time.monotonic()-start)
    dk.direct.save(_repo_io(out/'done.json'),r);print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','train']);p.add_argument('--tag',default='dino-kernel-f0');p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026101911);p.add_argument('--steps',type=int,default=60);p.add_argument('--batch',type=int,default=2);p.add_argument('--lr',type=float,default=.00003);p.add_argument('--anchor',type=float,default=.01);p.add_argument('--device',default='mps');a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):check(a) if a.action=='check' else train(a)
