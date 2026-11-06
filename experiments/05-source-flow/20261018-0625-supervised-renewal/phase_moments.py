"""Predict nonlinear and joint latent moments directly from original-image views.

Independent Gaussian geometry channels cannot recover cross-moments such as
E(C*S), E(phase*log-area), or E(darkness*log-area). Extra auxiliary targets below
are deterministic functions of permitted original-image weak annotations; no
rendered images or hardness pseudo-labels are introduced.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
import argparse,hashlib,json,math,resource,time
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
import phase_boundary_uncertain as b
p=b.p
KEYS=['logarea_squared','area_inverse_quarter','logar','logar_squared','logar_squared_cos4','logar_squared_sin4','darkness_squared','darkness_logarea','darkness_logar','phase_logarea','phase_logar','phase_darkness']

def moments(r,t):
    la,c,s,z=r.unbind(1);ar=(c.square()+s.square()+1e-10).sqrt();phase=(t==1).to(r.dtype)
    return torch.stack([la.square(),torch.exp(-la.clamp(0,14)/4),ar,ar.square(),c.square()-s.square(),2*c*s,z.square(),z*la,z*ar,phase*la,phase*ar,phase*z],1)

def scales(aux):
    r=torch.tensor(aux['reg']);t=torch.tensor(aux['classes']);v=torch.tensor(aux['valid']);s=torch.zeros(12,dtype=torch.float64);ss=s.clone();n=0
    for k in range(0,len(r),8):
        a=moments(r[k:k+8],t[k:k+8]).double();mask=v[k:k+8,None];s+=(a*mask).sum((0,2,3));ss+=(a.square()*mask).sum((0,2,3));n+=int(mask.sum())
    mu=s/n;sd=(ss/n-mu.square()).clamp(min=1e-4).sqrt();return mu.numpy(),sd.numpy()

def extract_features(o,aux):
    base=b.BASE_FEATURES(o[:,:8],aux);pr=o[:,:4].softmax(1);g=pr[:,0]+pr[:,1];mass=g.sum((1,2)).clamp(min=1e-6)
    m=o[:,8:20]*p.tensor(aux['moment_scale'])[None,:,None,None]+p.tensor(aux['moment_mean'])[None,:,None,None]
    means=(m*g[:,None]).sum((2,3))/mass[:,None]
    # Raw direct moment predictions remain available alongside two moment variances.
    derived=torch.stack([(means[:,0]-base[:,3].square()).clamp(min=0).sqrt(),(means[:,6]-base[:,10].square()).clamp(min=0).sqrt(),means[:,7]-base[:,10]*base[:,3],means[:,8]-base[:,10]*means[:,2]],1)
    return torch.cat([base,means,derived],1)

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(.35)
    parent=_repo_path(HERE, 'runs', a.parent);out=_repo_path(HERE, 'runs', a.tag);out.mkdir(exist_ok=True)
    if (out/'done.json').exists():return
    d=dict(np.load(_repo_path(HERE, 'data.npz')));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');fc=np.load(_repo_io(parent/'frozen-features.npz'));tr=fc['train'];q=fc['query'];aux=p.prepare(d,px,tr,parent);assert not set(aux['indices'])&set(q);aux['moment_mean'],aux['moment_scale']=scales(aux)
    ck=torch.load(_repo_io(parent/'pretrained.pt'),map_location='cpu',weights_only=True);assert ck['train']==tr.tolist();torch.manual_seed(2026101889);torch.mps.manual_seed(2026101889);net=p.UNet(32,nout=20);state=net.state_dict()
    for k,v in ck['state'].items():
        if k in ['out.weight','out.bias']:state[k][:8]=v;state[k][8:]=0
        else:state[k]=v
    net.load_state_dict(state);net.to('mps');opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.0001);rng=np.random.default_rng(2026101889)
    spec=vars(a)|dict(seed=2026101889,train=tr.tolist(),query=q.tolist(),train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist(),moment_keys=KEYS,moment_mean=aux['moment_mean'].tolist(),moment_scale=aux['moment_scale'].tolist(),
        objective='CE+0.5normalizedgeometryMSE+0.5normalizeddirectmomentMSE,validweakgraininteriorsonly. Originalsame-imageboundaryweakeningandordinarytransforms; no augmentedhardnesslabels.',
        hypothesis='Estimate nonlinear/joint geometry moments directly instead of multiplying conditional means or assuming independent Gaussian latent variables. Weak annotations remain proxies, not physical truth.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),augmentation_source_sha256=hashlib.sha256((_repo_path(HERE, 'phase_boundary_uncertain.py')).read_bytes()).hexdigest(),parent_sha256=hashlib.sha256((parent/'pretrained.pt').read_bytes()).hexdigest())
    p.ga.save(_repo_io(out/'recipe.json'),spec);first=0;curve=[];start=time.monotonic()
    if (out/'resume.pt').exists():
        r=torch.load(_repo_io(out/'resume.pt'),map_location='cpu',weights_only=False);assert r['recipe']==spec;net.load_state_dict(r['state']);opt.load_state_dict(r['optimizer']);rng.bit_generator.state=r['rng'];b.WEAK_RNG.bit_generator.state=r['weak_rng'];first=r['step'];curve=r['curve']
        for s in opt.state.values():
            for k,v in s.items():
                if torch.is_tensor(v) and k!='step':s[k]=v.to('mps')
    for step in range(first,a.steps):
        x,t,r,v=b.boundary_batch(aux,px,rng,batch=4);net.train();o=net(x);reg=(o[:,4:8]-(r-p.tensor(aux['mean'])[None,:,None,None])/p.tensor(aux['scale'])[None,:,None,None]).square().mean(1)
        target=(moments(r,t)-p.tensor(aux['moment_mean'])[None,:,None,None])/p.tensor(aux['moment_scale'])[None,:,None,None];me=(o[:,8:20]-target).square().mean(1);loss=F.cross_entropy(o[:,:4],t)+.5*((reg+me)*v).sum()/v.sum().clamp(min=1);assert torch.isfinite(loss)
        for g in opt.param_groups:g['lr']=.0001*(.05+.95*.5*(1+math.cos(math.pi*step/a.steps)))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        if (step+1)%25==0:torch.mps.synchronize();torch.mps.empty_cache()
        if (step+1)%100==0:
            mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+torch.mps.driver_allocated_memory();assert mem<6*2**30;row=dict(step=step+1,loss=float(loss.detach().cpu()),seconds=time.monotonic()-start,memory=mem);curve.append(row);print(json.dumps(row),flush=True)
        if (step+1)%250==0 or step+1==a.steps:
            r=dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),recipe=spec,step=step+1,curve=curve,rng=rng.bit_generator.state,weak_rng=b.WEAK_RNG.bit_generator.state);torch.save(r,_repo_io(out/'resume-new.pt'));(out/'resume-new.pt').replace(out/'resume.pt')
    torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},recipe=spec,aux_normalization={k:aux[k] for k in ['mean','scale','moment_mean','moment_scale']}),_repo_io(out/'model.pt'));p.physical_features=extract_features
    f=p.extract(net,px,aux,tr,batch=4);v=p.extract(net,px,aux,q,batch=4);np.savez(_repo_io(out/'frozen-features.npz'),train=tr,query=q,train_features=f,query_features=v,train_globals=fc['train_globals'],query_globals=fc['query_globals'])
    p.ga.save(_repo_io(out/'done.json'),dict(complete=True,feature_dimension=int(f.shape[1]),seconds=time.monotonic()-start))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--parent',default='phase-joint-b4-f0-s81');ap.add_argument('--tag',default='phase-moments-f0-s89');ap.add_argument('--steps',type=int,default=2000)
    with threadpool_limits(2):main(ap.parse_args())
