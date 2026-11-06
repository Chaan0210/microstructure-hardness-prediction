"""Supervised adaptation of the last two eligible Cellpose-SAM encoder blocks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'));os.environ['HF_HUB_OFFLINE']='1'
import argparse,functools,hashlib,json,math,resource,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from segment_anything.modeling.image_encoder import Block
from segment_anything.modeling.common import LayerNorm2d

def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def read_prefix(path,indices,views):
    mapped=np.load(_repo_io(path),mmap_mode='r')
    batch=np.array(mapped[indices,views],dtype=np.float32)
    del mapped
    return batch

class Tail(nn.Module):
    def __init__(self,initial):
        super().__init__();self.blocks=nn.ModuleList()
        for j in range(2):
            size=(initial[f'blocks.{j}.attn.rel_pos_h'].shape[0]+1)//2
            self.blocks.append(Block(dim=1024,num_heads=16,mlp_ratio=4,qkv_bias=True,
                norm_layer=functools.partial(nn.LayerNorm,eps=1e-6),use_rel_pos=True,window_size=0,input_size=(size,size)))
        self.neck=nn.Sequential(nn.Conv2d(1024,256,1,bias=False),LayerNorm2d(256),nn.Conv2d(256,256,3,padding=1,bias=False),LayerNorm2d(256))
        self.load_state_dict(initial,strict=True)
    def spatial(self,x):
        for block in self.blocks:x=block(x)
        return self.neck(x.permute(0,3,1,2))
    def forward(self,x):
        z=self.spatial(x)
        return torch.cat([z.mean((2,3)),(z.var((2,3),correction=0)+1e-6).sqrt()],1)

class Regressor(nn.Module):
    def __init__(self,initial,pooling='global'):
        super().__init__();self.body=Tail(initial);self.pooling=pooling
        dim=512
        if pooling=='conditional':
            self.fuse=nn.Sequential(nn.Conv2d(260,64,1),nn.SiLU(),nn.Conv2d(64,64,3,padding=1),nn.SiLU());dim=128
        self.project=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,128),nn.SiLU(),nn.Dropout(.2))
        self.head=nn.Sequential(nn.Linear(132,64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,1))
        nn.init.normal_(self.head[-1].weight,std=.01);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,raw,patch=None):
        if self.pooling=='conditional':
            z=self.fuse(torch.cat([self.body.spatial(x),patch],1))
            feat=torch.cat([z.mean((2,3)),(z.var((2,3),correction=0)+1e-6).sqrt()],1)
        else:feat=self.body(x)
        return self.head(torch.cat([self.project(feat),raw],1)).squeeze(1)

def patch_channels(pixels):
    """Co-located original intensity and quality; no learned cross-image fitting."""
    out=[]
    for im in pixels[:,0]:
        x=im.astype(np.float32)/255.;mu=x.reshape(32,8,32,8).mean((1,3));var=(x*x).reshape(32,8,32,8).mean((1,3))-mu*mu
        lo,med,hi=np.percentile(x,[5,50,95]);scale=max(float(hi-lo),.01)
        hh=(x[::2,::2]-x[::2,1::2]-x[1::2,::2]+x[1::2,1::2])/2
        noise=float(np.median(abs(hh))/.6745)
        p=np.stack([(mu-.5)*4,(mu-float(med))/scale,np.sqrt(np.maximum(var,0))*4,np.full_like(mu,noise*4)])
        views=[]
        for rot,flip in [(0,False),(1,False),(0,True),(1,True)]:
            z=np.rot90(p,rot,axes=(1,2));views.append(np.ascontiguousarray(z[:,:,::-1] if flip else z))
        out.append(views)
    return np.asarray(out,dtype=np.float32)

@torch.no_grad()
def predict(model,features,raw,patches,ix,mu,sd,batch,views=4):
    model.eval();out=[]
    for first in range(0,len(ix),batch):
        jj=ix[first:first+batch];p=[];r=torch.tensor(raw[jj],dtype=torch.float32,device='mps')
        for view in range(views):
            x=torch.tensor(read_prefix(features,jj,view),device='mps')
            patch=torch.tensor(patches[jj,view],device='mps') if patches is not None else None
            p.append(model(x,r,patch).cpu().numpy())
        out.extend(np.mean(p,axis=0)*sd+mu)
    return np.array(out,dtype=float)

def main(a):
    assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.35)
    d=np.load(_repo_path(HERE, 'data.npz'));meta=json.loads((_repo_path(HERE, 'cpsam-prefix/meta.json')).read_text());assert meta['ids']==d['ids'].tolist()
    assert np.load(_repo_path(HERE, 'cpsam-prefix/done.npy')).all()
    features=_repo_path(HERE, 'cpsam-prefix/features.npy')
    if a.split=='historical':tr=d['dev'][d['fold']!=a.fold];va=d['dev'][d['fold']==a.fold]
    else:
        from sklearn.model_selection import KFold
        tr,va=list(KFold(5,shuffle=True,random_state=a.split_seed).split(np.arange(500)))[a.fold]
    assert set(tr).isdisjoint(va)
    px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');raw=[]
    for x in px:
        z=(x.astype(float)/255-.5)*4;raw.append([float(np.mean(z**k)) for k in [1,2,3,4]])
    raw=np.array(raw);rmu=raw[tr].mean(0);rsd=np.maximum(raw[tr].std(0),1e-6);raw=((raw-rmu)/rsd).astype(np.float32)
    patches=patch_channels(px) if a.pooling=='conditional' else None
    mu=float(d['y'][tr].mean());sd=float(d['y'][tr].std());yn=((d['y']-mu)/sd).astype(np.float32)
    folder=_repo_path(HERE, 'runs', a.tag);folder.mkdir(parents=True,exist_ok=True);assert not (folder/'done.json').exists()
    recipe=vars(a)|dict(train=tr.tolist(),validation=va.tolist(),train_ids=d['ids'][tr].tolist(),validation_ids=d['ids'][va].tolist(),
        target_mean=mu,target_std=sd,raw_mean=rmu.tolist(),raw_std=rsd.tolist(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prefix_recipe=meta,torch=str(torch.__version__))
    if (folder/'recipe.json').exists():assert json.loads((folder/'recipe.json').read_text())==recipe
    else:save(folder/'recipe.json',recipe)
    torch.manual_seed(a.seed);torch.mps.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    initial=torch.load(_repo_path(HERE, 'cpsam-tail-initial.pt'),map_location='cpu',weights_only=True);model=Regressor(initial,a.pooling).to('mps');del initial
    body_ids={id(p) for p in model.body.parameters()};head_params=[p for p in model.parameters() if id(p) not in body_ids]
    opt=torch.optim.AdamW([dict(params=model.body.parameters(),lr=a.body_lr),dict(params=head_params,lr=a.head_lr)],weight_decay=.001)
    start=time.monotonic();curve=[];avg=None;avg_n=0;first_epoch=0;elapsed=0;resume=folder/'resume.pt'
    if resume.exists():
        ck=torch.load(_repo_io(resume),map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        for state in opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.to('mps')
        first_epoch=ck['epoch'];curve=ck['curve'];avg=ck['average'];avg_n=ck['average_n'];elapsed=ck['elapsed']
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng']);torch.mps.set_rng_state(ck['mps_rng'])
    print('START',a.tag,'train',len(tr),'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(first_epoch,a.epochs):
        adapt=not a.frozen and epoch>=a.warmup
        for p in model.body.parameters():p.requires_grad_(adapt)
        factor=.05+.95*.5*(1+math.cos(math.pi*epoch/a.epochs))
        opt.param_groups[0]['lr']=a.body_lr*factor if adapt else 0
        opt.param_groups[1]['lr']=a.head_lr*factor*min(1.,(epoch+1)/3)
        model.train();order=rng.permutation(tr);total=0
        for first in range(0,len(order),a.batch):
            ix=order[first:first+a.batch];views=rng.integers(4,size=len(ix))
            x=torch.tensor(read_prefix(features,ix,views),device='mps')
            r=torch.tensor(raw[ix],device='mps');y=torch.tensor(yn[ix],device='mps')
            patch=torch.tensor(patches[ix,views],device='mps') if patches is not None else None
            pred=model(x,r,patch);loss=F.mse_loss(pred,y);assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach().cpu())*len(ix)
        if epoch>=a.epochs-5:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};avg_n+=1
            if avg is None:avg=state
            else:
                for k,v in state.items():avg[k].add_((v-avg[k])/avg_n)
        rec=dict(epoch=epoch+1,adapt=adapt,online_train_rmse=float(np.sqrt(total/len(tr))*sd),elapsed=elapsed+time.monotonic()-start)
        if epoch==0 or (epoch+1)%a.eval_every==0 or epoch+1==a.epochs:
            vp=predict(model,features,raw,patches,va,mu,sd,a.batch);tp=predict(model,features,raw,patches,tr[:64],mu,sd,a.batch,views=1)
            rec.update(validation_rmse=rmse(vp,d['y'][va]),train64_rmse=rmse(tp,d['y'][tr[:64]]))
            np.savez(_repo_io(folder/f'epoch{epoch+1:03d}-predictions.npz'),query=va,prediction=vp);print(json.dumps(rec),flush=True)
        curve.append(rec)
        if (epoch+1)%a.eval_every==0 or epoch+1==a.epochs:
            ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),epoch=epoch+1,curve=curve,average=avg,average_n=avg_n,
                elapsed=rec['elapsed'],rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state())
            torch.save(ck,_repo_io(folder/'resume-new.pt'));(folder/'resume-new.pt').replace(resume);save(folder/'curve.json',curve)
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+torch.mps.driver_allocated_memory()
        if mem>6*2**30:raise RuntimeError(f'Own memory budget exceeded: {mem}')
    model.load_state_dict(avg);vp=predict(model,features,raw,patches,va,mu,sd,a.batch);tp=predict(model,features,raw,patches,tr,mu,sd,a.batch)
    torch.save(dict(state=avg,recipe=recipe),_repo_io(folder/'model.pt'));np.savez(_repo_io(folder/'predictions.npz'),train=tr,query=va,prediction=vp,train_prediction=tp,ids=d['ids'][va])
    result=dict(train_rmse=rmse(tp,d['y'][tr]),validation_rmse=rmse(vp,d['y'][va]),seconds=elapsed+time.monotonic()-start)
    if a.split=='historical':
        base=d['base083'][d['fold']==a.fold];result.update(base083_rmse=rmse(base,d['y'][va]),fixed_quarter_blend_rmse=rmse(.75*base+.25*vp,d['y'][va]),error_correlation=float(np.corrcoef(vp-d['y'][va],base-d['y'][va])[0,1]))
    save(folder/'done.json',result);print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026101811)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--frozen',action='store_true');p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--body-lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=.001);p.add_argument('--batch',type=int,default=4);p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--pooling',choices=['global','conditional'],default='global')
    p.add_argument('--split',choices=['historical','confirmation'],default='historical');p.add_argument('--split-seed',type=int,default=2026101801);main(p.parse_args())
