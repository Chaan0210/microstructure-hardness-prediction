"""Partition-local phase/geometry auxiliary training followed by joint hardness fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'))
import argparse,copy,gc,hashlib,json,math,resource,time
import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from phase_components import UNet,targets
import grain_attention_v2 as ga

def norm(a):
    lo,med,hi=np.percentile(a,[5,50,95]);return ((a-float(med))/max(float(hi-lo),1.)).astype(np.float32)
def transform(a,rot,flip=False):
    z=np.rot90(a,rot,axes=(-2,-1));return np.ascontiguousarray(z[...,::-1] if flip else z)
def tensor(a,dtype=torch.float32):return torch.tensor(a,dtype=dtype,device='mps')
def memory():
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;driver=torch.mps.driver_allocated_memory();m=rss+driver
    if m>6*2**30:raise RuntimeError(f'Own memory budget exceeded: total={m}, peakRSS={rss}, MPSdriver={driver}')
    return m

def prepare(d,px,tr,out):
    path=out/'auxiliary.npz'
    if path.exists():
        a=dict(np.load(_repo_io(path)));assert np.array_equal(a['train'],tr)
        mp=out/'auxiliary-masks.npy'
        if not mp.exists():
            masks=np.load(_repo_path(ROOT, 'experiments/grain-cellpose-220/masks-v3-train.npz'));mi={str(v):j for j,v in enumerate(masks['ids'])}
            np.save(_repo_io(mp),np.array([masks['masks'][mi[str(d['ids'][j])]] for j in a['indices']],dtype=np.uint16))
        a['masks']=np.load(_repo_io(mp),mmap_mode='r');return a
    masks=np.load(_repo_path(ROOT, 'experiments/grain-cellpose-220/masks-v3-train.npz'));mi={str(v):j for j,v in enumerate(masks['ids'])}
    ix=[];classes=[];reg=[];valid=[]
    for j in tr:
        if d['noise'][j]>=9.5:continue
        m=masks['masks'][mi[str(d['ids'][j])]]
        if m.max()<20:continue
        t,r,v=targets(px[j,0],m)
        if not v.any():continue
        ix.append(j);classes.append(t.astype(np.uint8));reg.append(r);valid.append(v)
    rr=np.array(reg);vv=np.array(valid)
    la_mu=float(np.mean([r[0][v].mean() for r,v in zip(rr,vv)]));la_sd=float(np.mean([r[0][v].std() for r,v in zip(rr,vv)])+.5)
    a=dict(train=tr,indices=np.array(ix),classes=np.array(classes),reg=rr,valid=vv,mean=np.array([la_mu,0,0,.15],np.float32),scale=np.array([la_sd,.3,.3,.15],np.float32))
    np.savez_compressed(_repo_io(path),**a)
    np.save(_repo_io(out/'auxiliary-masks.npy'),np.array([masks['masks'][mi[str(d['ids'][j])]] for j in a['indices']],dtype=np.uint16))
    a['masks']=np.load(_repo_io(out/'auxiliary-masks.npy'),mmap_mode='r');return a

def aux_batch(aux,px,rng,batch=8):
    chosen=rng.integers(len(aux['indices']),size=batch);xs=[];tt=[];rr=[];vv=[]
    for q in chosen:
        j=aux['indices'][q];im=px[j,0].astype(np.float32)
        # Ordinary transformations of this original image. No reconstructed layout.
        if rng.random()<.85:
            labels=aux['masks'][q];offset=rng.normal(0,rng.uniform(0,4),int(labels.max())+1);offset[0]=0
            im=im+offset[labels]  # Local brightness shifts of original pixels, without redrawing grains.
            sigma=float(rng.uniform(0,1.4));im=ndi.gaussian_filter(im,sigma) if sigma>.05 else im
            mean=float(im.mean());im=mean+float(rng.uniform(.7,1.05))*(im-mean)+float(rng.uniform(-10,10))
            im=im+rng.normal(0,rng.uniform(0,20),im.shape)
        im=np.clip(im,0,255);rot=int(rng.integers(4));flip=bool(rng.integers(2))
        r=transform(aux['reg'][q],rot,flip).copy();r[1:3]*=(-1 if rot%2 else 1)
        if flip:r[2]*=-1
        xs.append(transform(norm(im),rot,flip));tt.append(transform(aux['classes'][q],rot,flip));rr.append(r);vv.append(transform(aux['valid'][q],rot,flip))
    return tensor(np.array(xs)[:,None]),tensor(np.array(tt),torch.long),tensor(np.array(rr)),tensor(np.array(vv))

def aux_loss(net,pack,aux):
    x,t,r,v=pack;o=net(x);ce=F.cross_entropy(o[:,:4],t)
    mu=tensor(aux['mean'])[None,:,None,None];sd=tensor(aux['scale'])[None,:,None,None]
    loss=(((o[:,4:]-(r-mu)/sd)**2).mean(1)*v).sum()/v.sum().clamp(min=1)
    return ce+.5*loss

def pretrain(aux,px,a,out):
    torch.manual_seed(a.seed);torch.mps.manual_seed(a.seed);rng=np.random.default_rng(a.seed);net=UNet(a.width).to('mps')
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);start=time.monotonic();first=0;curve=[]
    ckpath=out/'pretrain-resume.pt'
    if ckpath.exists():
        ck=torch.load(_repo_io(ckpath),map_location='cpu',weights_only=False);assert ck['target_steps']==a.pretrain_steps and ck['width']==a.width and ck['seed']==a.seed and ck['pretrain_batch']==a.pretrain_batch
        net.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);first=ck['step'];curve=ck['curve'];rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng']);torch.mps.set_rng_state(ck['mps_rng'])
        for s in opt.state.values():
            for k,v in s.items():
                if torch.is_tensor(v) and k!='step':s[k]=v.to('mps')
    for step in range(first,a.pretrain_steps):
        lr=.001*min(1.,(step+1)/100)*(.05+.95*.5*(1+math.cos(math.pi*step/a.pretrain_steps)))
        for g in opt.param_groups:g['lr']=lr
        net.train();loss=aux_loss(net,aux_batch(aux,px,rng,batch=a.pretrain_batch),aux);assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        if (step+1)%25==0:torch.mps.synchronize();torch.mps.empty_cache()
        if (step+1)%100==0:
            rec=dict(step=step+1,loss=float(loss.detach().cpu()),seconds=time.monotonic()-start,memory=memory());curve.append(rec);print('AUX',json.dumps(rec),flush=True)
        if (step+1)%250==0 or step+1==a.pretrain_steps:
            ck=dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),step=step+1,curve=curve,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),target_steps=a.pretrain_steps,width=a.width,seed=a.seed,pretrain_batch=a.pretrain_batch)
            torch.save(ck,_repo_io(out/'pretrain-new.pt'));(out/'pretrain-new.pt').replace(ckpath)
    torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},train=aux['train'].tolist(),auxiliary_sources=aux['indices'].tolist(),normalization={k:aux[k].tolist() for k in ['mean','scale']},seed=a.seed,pretrain_steps=a.pretrain_steps,width=a.width,pretrain_batch=a.pretrain_batch),_repo_io(out/'pretrained.pt'))
    return net

def physical_features(o,aux):
    pr=o[:,:4].softmax(1);r=o[:,4:]*tensor(aux['scale'])[None,:,None,None]+tensor(aux['mean'])[None,:,None,None]
    g=pr[:,0]+pr[:,1];mass=g.sum((1,2)).clamp(min=1e-6)
    avg=lambda z:(g*z).sum((1,2))/mass
    phase=pr[:,1].sum((1,2))/mass;la=avg(r[:,0]);lasd=avg((r[:,0]-la[:,None,None])**2).clamp(min=1e-6).sqrt()
    C=avg(r[:,1]);S=avg(r[:,2]);mag=(C*C+S*S+1e-6).sqrt();localmag=avg((r[:,1]**2+r[:,2]**2+1e-6).sqrt())
    ff=[phase,pr[:,2].mean((1,2)),pr[:,3].mean((1,2)),la,lasd,avg(torch.exp(-.25*r[:,0].clamp(-2,14))),mag,localmag,(C*C-S*S)/mag,2*C*S/mag,avg(r[:,3]),avg(torch.sigmoid((r[:,3]-.25)/.05)),avg(torch.sigmoid((r[:,3]-.5)/.05))]
    for k in [16,32,64]:
        ph=F.avg_pool2d(pr[:,1:2],k)/F.avg_pool2d(g[:,None],k).clamp(min=1e-6)
        ff.append((ph.var((1,2,3),correction=0)+1e-6).sqrt())
        ar=F.avg_pool2d((g*r[:,0])[:,None],k)/F.avg_pool2d(g[:,None],k).clamp(min=1e-6)
        ff.append((ar.var((1,2,3),correction=0)+1e-6).sqrt())
    return torch.stack(ff,1)

@torch.no_grad()
def extract(net,px,aux,indices,batch=8):
    net.eval();rows=[]
    for first in range(0,len(indices),batch):
        xx=np.array([norm(px[j,0]) for j in indices[first:first+batch]])[:,None];acc=[]
        for rot in [0,1,2,3]:acc.append(physical_features(net(tensor(transform(xx,rot))),aux).cpu().numpy())
        rows.append(np.mean(acc,0))
    return np.concatenate(rows)

class Joint(nn.Module):
    def __init__(self,net,aux,mean,scale,coef,bias):
        super().__init__();self.net=net;self.aux=aux
        self.register_buffer('mean',tensor(mean));self.register_buffer('scale',tensor(scale))
        self.head=nn.Linear(len(mean),1).to('mps')
        with torch.no_grad():self.head.weight.copy_(tensor(coef[None]));self.head.bias.fill_(float(bias))
    def forward(self,x,g):
        f=torch.cat([physical_features(self.net(x),self.aux),g],1)
        return self.head((f-self.mean)/self.scale)[:,0]

@torch.no_grad()
def predict(model,px,G,indices,ym,ys,batch=8):
    model.eval();out=[]
    for first in range(0,len(indices),batch):
        ix=indices[first:first+batch];x=np.array([norm(px[j,0]) for j in ix])[:,None];g=tensor(G[ix]);ps=[]
        for rot in [0,1,2,3]:ps.append(model(tensor(transform(x,rot)),g).cpu().numpy())
        out.extend(np.mean(ps,0)*ys+ym)
    return np.array(out,dtype=float)

def main(a):
    assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.35)
    d=dict(np.load(_repo_path(HERE, 'data.npz')));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');tr=d['dev'][d['fold']!=a.fold];va=d['dev'][d['fold']==a.fold]
    out=_repo_path(HERE, 'runs', a.tag);out.mkdir(parents=True,exist_ok=True);assert not (out/'done.json').exists()
    aux=prepare(d,px,tr,out)
    recipe=vars(a)|dict(train=tr.tolist(),validation=va.tolist(),train_ids=d['ids'][tr].tolist(),validation_ids=d['ids'][va].tolist(),auxiliary_ids=d['ids'][aux['indices']].tolist(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),weak_target_source_sha256=hashlib.sha256((_repo_path(HERE, 'phase_components.py')).read_bytes()).hexdigest(),
        aux_target='Original train-only zero-shot mask classes and geometry; not physical truth. Unweighted CE + .5 normalized geometry MSE.',
        augmentation='Original pixels: per-grain brightness offsets withSD0–4gray,Gaussian blur0–1.4,contrast.7–1.05,globalbrightness±10,noise0–20gray,D4covariant auxiliary targets. No new layouts/boundary replacement. Hardness training original+C4 only.',
        objective='Direct hardness MSE plus.25 auxiliary loss and.02 head-parameter anchor; no teacher predictions or fixed residual targets.',ridge_alpha=10.,BN='pretrained running statistics frozen during hardness fine-tuning')
    ga.save(_repo_io(out/'recipe.json'),recipe)
    if (out/'pretrained.pt').exists():
        ck=torch.load(_repo_io(out/'pretrained.pt'),map_location='cpu',weights_only=True);assert ck['train']==tr.tolist() and ck['pretrain_steps']==a.pretrain_steps and ck['width']==a.width and ck['seed']==a.seed and ck['pretrain_batch']==a.pretrain_batch
        net=UNet(a.width).to('mps');net.load_state_dict(ck['state'])
    else:net=pretrain(aux,px,a,out)
    if a.pretrain_only:return
    _,_,G,gd,prov=ga.load();b=np.load(_repo_path(ROOT, 'experiments/probabilistic-grains-900/object-bank-train.npz'));ok=list(b['object_keys']);gk=list(b['global_keys'])
    names=[f'mean_{k}' for k in ok]+[f'sd_{k}' for k in ok]+gk;drop={f'{s}_{k}' for s in ['mean','sd'] for k in ['cx','cy','cos2','sin2']}|{'orient_c2','orient_s2'};G=G[:,[n not in drop for n in names]]
    f=extract(net,px,aux,tr);X=np.c_[f,G[tr]];mean=X.mean(0);scale=np.maximum(X.std(0),.01);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());target=(d['y'][tr]-ym)/ys
    ridge=Ridge(alpha=10.).fit((X-mean)/scale,target)
    model=Joint(net,aux,mean,scale,ridge.coef_,ridge.intercept_);frozen=predict(model,px,G,va,ym,ys)
    np.savez(_repo_io(out/'frozen-features.npz'),train=tr,query=va,train_features=f,query_features=extract(net,px,aux,va),train_globals=G[tr],query_globals=G[va])
    np.savez(_repo_io(out/'frozen-predictions.npz'),train=tr,query=va,prediction=frozen)
    print('FROZEN',ga.rmse(frozen,d['y'][va]),flush=True)
    initial_head=model.head.weight.detach().clone();initial_bias=model.head.bias.detach().clone()
    opt=torch.optim.AdamW([dict(params=model.net.parameters(),lr=.0001),dict(params=model.head.parameters(),lr=.001)],weight_decay=.0001)
    torch.manual_seed(a.seed+17);torch.mps.manual_seed(a.seed+17);rng=np.random.default_rng(a.seed+17);start=time.monotonic();curve=[];average=None;navg=0;first_epoch=0;elapsed=0.
    if (out/'hardness-checkpoint.pt').exists():
        ck=torch.load(_repo_io(out/'hardness-checkpoint.pt'),map_location='cpu',weights_only=False);assert ck['recipe']==recipe
        model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer'])
        for state in opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value) and key!='step':state[key]=value.to('mps')
        first_epoch=ck['epoch'];curve=ck['curve'];average=ck['average'];navg=ck['navg'];elapsed=ck['elapsed'];rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng']);torch.mps.set_rng_state(ck['mps_rng'])
    for epoch in range(first_epoch,a.epochs):
        factor=.05+.95*.5*(1+math.cos(math.pi*epoch/a.epochs));opt.param_groups[0]['lr']=.0001*factor;opt.param_groups[1]['lr']=.001*factor
        model.train()
        for layer in model.net.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        order=rng.permutation(tr);hard_sum=aux_sum=0.
        for first in range(0,len(tr),a.batch):
            ix=order[first:first+a.batch];rot=int(rng.integers(4));x=np.array([norm(px[j,0]) for j in ix])[:,None]
            p=model(tensor(transform(x,rot)),tensor(G[ix]));y=tensor((d['y'][ix]-ym)/ys);mse=F.mse_loss(p,y)
            al=aux_loss(model.net,aux_batch(aux,px,rng,batch=a.aux_batch),aux)
            anchor=.02*((model.head.weight-initial_head).square().sum()+(model.head.bias-initial_bias).square().sum())
            loss=mse+a.aux_weight*al+anchor;assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            hard_sum+=float(mse.detach().cpu())*len(ix);aux_sum+=float(al.detach().cpu())*len(ix)
            if (first//a.batch+1)%25==0:torch.mps.synchronize();torch.mps.empty_cache()
        if epoch>=a.epochs-5:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};navg+=1
            if average is None:average=state
            else:
                for k,v in state.items():
                    if v.is_floating_point():average[k].add_((v-average[k])/navg)
                    else:average[k]=v
        rec=dict(epoch=epoch+1,train_online_rmse=float(np.sqrt(hard_sum/len(tr))*ys),auxiliary_loss=aux_sum/len(tr),seconds=elapsed+time.monotonic()-start,memory=memory())
        if epoch==0 or (epoch+1)%10==0:
            vp=predict(model,px,G,va,ym,ys);rec['validation_rmse']=ga.rmse(vp,d['y'][va]);np.savez(_repo_io(out/f'epoch{epoch+1:03d}.npz'),query=va,prediction=vp)
            torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),epoch=epoch+1,recipe=recipe,mean=mean.tolist(),scale=scale.tolist(),ym=ym,ys=ys,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),curve=curve+[rec],average=average,navg=navg,elapsed=rec['seconds']),_repo_io(out/'hardness-new.pt'))
            (out/'hardness-new.pt').replace(out/'hardness-checkpoint.pt')
            print('JOINT',json.dumps(rec),flush=True)
        curve.append(rec);ga.save(_repo_io(out/'hardness-curve.json'),curve)
    model.load_state_dict(average);vp=predict(model,px,G,va,ym,ys);tp=predict(model,px,G,tr,ym,ys)
    np.savez(_repo_io(out/'joint-features.npz'),train=tr,query=va,train_features=extract(model.net,px,aux,tr),query_features=extract(model.net,px,aux,va),train_globals=G[tr],query_globals=G[va])
    torch.save(dict(state=average,recipe=recipe,mean=mean.tolist(),scale=scale.tolist(),ym=ym,ys=ys,auxiliary_normalization={k:aux[k].tolist() for k in ['mean','scale']},global_columns=[n for n in names if n not in drop]),_repo_io(out/'model.pt'))
    np.savez(_repo_io(out/'predictions.npz'),train=tr,query=va,prediction=vp,train_prediction=tp,frozen_prediction=frozen)
    mask=~d['failed'][d['fold']==a.fold];base=d['base083'][d['fold']==a.fold]
    result=dict(train_rmse=ga.rmse(tp,d['y'][tr]),rmse=ga.rmse(vp,d['y'][va]),frozen_rmse=ga.rmse(frozen,d['y'][va]),base083_rmse=ga.rmse(base,d['y'][va]),quarter_blend_rmse=ga.rmse(base+.25*mask*(vp-base),d['y'][va]),ungated_quarter_blend_rmse=ga.rmse(.75*base+.25*vp,d['y'][va]),fine_tune_seconds=elapsed+time.monotonic()-start)
    ga.save(_repo_io(out/'done.json'),result);print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026101881)
    p.add_argument('--width',type=int,default=32);p.add_argument('--pretrain-steps',type=int,default=4000);p.add_argument('--pretrain-batch',type=int,default=4);p.add_argument('--epochs',type=int,default=80);p.add_argument('--batch',type=int,default=4);p.add_argument('--aux-batch',type=int,default=2);p.add_argument('--aux-weight',type=float,default=.25);p.add_argument('--pretrain-only',action='store_true')
    with threadpool_limits(limits=2):main(p.parse_args())
