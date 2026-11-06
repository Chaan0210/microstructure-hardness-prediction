"""Mean-only versus conditional-variance auxiliary training, same originals and visits."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'))
import argparse,hashlib,json,math,resource,time
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
import phase_joint_v3 as p
from scipy import ndimage as ndi

BASE_FEATURES=p.physical_features
BASE_BATCH=p.aux_batch
WEAK_ENABLED=True
WEAK_RNG=np.random.default_rng(2026101941)
NEIGHBORS={}

def boundary_batch(aux,px,rng,batch=8):
    chosen=rng.integers(len(aux['indices']),size=batch);xs=[];tt=[];rr=[];vv=[]
    for q in chosen:
        j=aux['indices'][q];im=px[j,0].astype(np.float32)
        if rng.random()<.85:
            labels=aux['masks'][q];offset=rng.normal(0,rng.uniform(0,4),int(labels.max())+1);offset[0]=0
            sigma=float(rng.uniform(0,1.4));contrast=float(rng.uniform(.7,1.05));shift=float(rng.uniform(-10,10));sn=float(rng.uniform(0,20));noise=rng.normal(0,sn,im.shape)
            im=im+offset[labels]
            if WEAK_ENABLED and sn>=8:
                if int(q) not in NEIGHBORS:
                    inside=(ndi.minimum_filter(labels,3)==ndi.maximum_filter(labels,3))&(labels>0)
                    distance,near=ndi.distance_transform_edt(~inside,return_indices=True)
                    same=labels[near[0],near[1]]==labels
                    dest=np.flatnonzero((~inside)&(labels>0)&(distance<=3)&same)
                    source=(near[0]*256+near[1]).ravel()[dest]
                    assert np.array_equal(labels.ravel()[dest],labels.ravel()[source])
                    NEIGHBORS[int(q)]=(dest,source)
                dest,source=NEIGHBORS[int(q)];prob=float(WEAK_RNG.uniform(.25,.65));selected=WEAK_RNG.random(int(labels.max())+1)<prob;strength=WEAK_RNG.uniform(.6,1.,int(labels.max())+1)
                use=selected[labels.ravel()[dest]];a=strength[labels.ravel()[dest[use]]];flat=im.ravel();flat[dest[use]]=(1-a)*flat[dest[use]]+a*flat[source[use]]
            im=ndi.gaussian_filter(im,sigma) if sigma>.05 else im
            mean=float(im.mean());im=mean+contrast*(im-mean)+shift;im=im+noise
        im=np.clip(im,0,255);rot=int(rng.integers(4));flip=bool(rng.integers(2))
        r=p.transform(aux['reg'][q],rot,flip).copy();r[1:3]*=(-1 if rot%2 else 1)
        if flip:r[2]*=-1
        xs.append(p.transform(p.norm(im),rot,flip));tt.append(p.transform(aux['classes'][q],rot,flip));rr.append(r);vv.append(p.transform(aux['valid'][q],rot,flip))
    return p.tensor(np.array(xs)[:,None]),p.tensor(np.array(tt),torch.long),p.tensor(np.array(rr)),p.tensor(np.array(vv))

def distribution_features(o,aux):
    base=BASE_FEATURES(o[:,:8],aux);pr=o[:,:4].softmax(1);g=pr[:,0]+pr[:,1];mass=g.sum((1,2)).clamp(min=1e-6)
    avg=lambda x:(g*x).sum((1,2))/mass
    scale=p.tensor(aux['scale'])[None,:,None,None];mean=o[:,4:8]*scale+p.tensor(aux['mean'])[None,:,None,None]
    variance=o[:,8:12].clamp(-5,2).exp()*scale.square()
    extra=[avg(variance[:,j]).clamp(min=1e-8).sqrt() for j in range(4)]
    extra.append(avg(torch.exp(-.25*mean[:,0].clamp(-2,14)+.03125*variance[:,0])))
    extra.append((avg(mean[:,0].square()+variance[:,0])-avg(mean[:,0]).square()).clamp(min=1e-8).sqrt())
    for threshold in [.25,.5]:extra.append(avg(.5*(1+torch.erf((mean[:,3]-threshold)/(2*variance[:,3]+1e-6).sqrt()))))
    extra.append(avg((mean[:,1].square()+mean[:,2].square()+variance[:,1]+variance[:,2]+1e-6).sqrt()))
    extra.append((avg(mean[:,3].square()+variance[:,3])-avg(mean[:,3]).square()).clamp(min=1e-8).sqrt())
    return torch.cat([base,torch.stack(extra,1)],1)

def objective(net,pack,aux,arm):
    x,t,r,v=pack;o=net(x);mu=p.tensor(aux['mean'])[None,:,None,None];sd=p.tensor(aux['scale'])[None,:,None,None]
    e=(o[:,4:8]-(r-mu)/sd).square()
    if arm=='nll':
        lv=o[:,8:12].clamp(-5,2);e=e*torch.exp(-lv)+lv
    reg=(e.mean(1)*v).sum()/v.sum().clamp(min=1)
    return F.cross_entropy(o[:,:4],t)+.5*reg

def run(a,arm):
    parent=_repo_path(HERE, 'runs/phase-joint-b4-f0-s81');out=_repo_path(HERE, 'runs', f'phase-boundary-{arm}-f0-s89');out.mkdir(exist_ok=True)
    if (out/'done.json').exists():return
    d=dict(np.load(_repo_path(HERE, 'data.npz')));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');fc=np.load(_repo_io(parent/'frozen-features.npz'));tr=fc['train'];q=fc['query'];assert not set(tr)&set(q)
    aux=p.prepare(d,px,tr,parent);ck=torch.load(_repo_io(parent/'pretrained.pt'),map_location='cpu',weights_only=True);assert ck['train']==tr.tolist()
    torch.manual_seed(2026101889)
    if a.device=='mps':torch.mps.manual_seed(2026101889)
    net=p.UNet(32,nout=12);state=net.state_dict()
    for key,value in ck['state'].items():
        if key in ['out.weight','out.bias']:state[key][:8]=value;state[key][8:]=0
        else:state[key]=value
    net.load_state_dict(state);net.to(a.device);rng=np.random.default_rng(2026101889);opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.0001)
    spec=dict(arm=arm,seed=2026101889,steps=a.steps,batch=4,device=a.device,train=tr.tolist(),validation=q.tolist(),auxiliary_sources=aux['indices'].tolist(),
        parent_checkpoint_sha256=hashlib.sha256((parent/'pretrained.pt').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hypothesis='Point estimates may lose nonlinear geometry moments under degradation. Learn conditional variances using original-derived weak targets and evaluate actual hardness.',
        interpretation='Variance/moment channels are model-derived descriptors. Pointwise Gaussian assumptions and weak masks do not establish calibrated physical uncertainty.',
        control='Existingphase-dist-nll-f0-s89:identical12channelinitialization,source/view/brightness/blur/noisestreams,2000updates. Extraindependentseed2026101941selectsboundaryweakening.',
        operation='Forviewswithaddednoise>=8gray,select25–65%ofinferredgrains;boundarypixelsblend60–100%withanearestinteriorpixelwithin3pxofthesamegraininthatoriginalimage,beforeblur/noise. No newlayout or separatelyrenderedimage.',
        target_assumption='Original-derivedweakgeometrytargetsremainlatenttargetswhenboundaryvisibilityisreduced;thisdoesnotestablishphysicaltruthorhardnessgain. No augmentedhardnesslabels are used.')
    p.ga.save(_repo_io(out/'recipe.json'),spec);start=time.monotonic();curve=[];first=0
    if (out/'resume.pt').exists():
        r=torch.load(_repo_io(out/'resume.pt'),map_location='cpu',weights_only=False);assert r['recipe']==spec;net.load_state_dict(r['state']);opt.load_state_dict(r['optimizer']);rng.bit_generator.state=r['rng'];WEAK_RNG.bit_generator.state=r['weak_rng'];first=r['step'];curve=r['curve']
        for s in opt.state.values():
            for k,v in s.items():
                if torch.is_tensor(v) and k!='step':s[k]=v.to(a.device)
    for step in range(first,a.steps):
        for g in opt.param_groups:g['lr']=.0001*(.05+.95*.5*(1+math.cos(math.pi*step/a.steps)))
        net.train();loss=objective(net,boundary_batch(aux,px,rng,batch=4),aux,arm);assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        if a.device=='mps' and (step+1)%25==0:torch.mps.synchronize();torch.mps.empty_cache()
        if (step+1)%100==0:
            mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+(torch.mps.driver_allocated_memory() if a.device=='mps' else 0);assert mem<6*2**30
            row=dict(step=step+1,loss=float(loss.detach().cpu()),seconds=time.monotonic()-start,memory=mem);curve.append(row);print(arm,json.dumps(row),flush=True)
        if (step+1)%250==0 or step+1==a.steps:
            r=dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),recipe=spec,step=step+1,curve=curve,rng=rng.bit_generator.state,weak_rng=WEAK_RNG.bit_generator.state)
            torch.save(r,_repo_io(out/'resume-new.pt'));(out/'resume-new.pt').replace(out/'resume.pt')
    torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},recipe=spec,normalization={k:aux[k].tolist() for k in ['mean','scale']}),_repo_io(out/'model.pt'))
    p.physical_features=distribution_features
    f=p.extract(net,px,aux,tr,batch=4);v=p.extract(net,px,aux,q,batch=4)
    np.savez(_repo_io(out/'frozen-features.npz'),train=tr,query=q,train_features=f,query_features=v,train_globals=fc['train_globals'],query_globals=fc['query_globals'])
    p.ga.save(_repo_io(out/'done.json'),dict(complete=True,steps=a.steps,feature_dimension=int(f.shape[1]),seconds=time.monotonic()-start,hardness_training=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','mps'],default='mps');parser.add_argument('--steps',type=int,default=2000);a=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.device=='mps':assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(.35)
    p.tensor=lambda x,dtype=torch.float32:torch.tensor(x,dtype=dtype,device=a.device)
    with threadpool_limits(limits=2):
        for arm in ['nll']:run(a,arm)
