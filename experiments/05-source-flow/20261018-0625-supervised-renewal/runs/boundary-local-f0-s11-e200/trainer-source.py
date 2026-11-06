"""Direct supervised image experiments. No test reads; every run is isolated."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, sys
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'));os.environ['MPLCONFIGDIR']=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import argparse, copy, hashlib, json, math, resource, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def save_json(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

class Block(nn.Module):
    def __init__(self,ci,co,down=False):
        super().__init__()
        self.pre=nn.AvgPool2d(2) if down else nn.Identity()
        self.skip=nn.Conv2d(ci,co,1,bias=False) if ci!=co else nn.Identity()
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.SiLU(),
                               nn.Conv2d(co,co,3,padding=1,bias=False),nn.BatchNorm2d(co))
        nn.init.zeros_(self.net[-1].weight)
    def forward(self,x):
        x=self.pre(x);return F.silu(self.skip(x)+self.net(x))

class Predictor(nn.Module):
    def __init__(self,kind='texture',width=16):
        super().__init__();self.kind=kind
        ch=[width,2*width,4*width,6*width]
        self.stem=nn.Sequential(nn.Conv2d(1,ch[0],5,padding=2,bias=False),nn.BatchNorm2d(ch[0]),nn.SiLU(),nn.AvgPool2d(2))
        self.stages=nn.ModuleList([nn.Sequential(Block(ch[0],ch[0]),Block(ch[0],ch[0]))]+[
            nn.Sequential(Block(ci,co,True),Block(co,co)) for ci,co in zip(ch[:-1],ch[1:])])
        self.projections=nn.ModuleList([nn.Conv2d(c,8,1) for c in ch])
        size=4+2*sum(ch)+4*64
        if kind=='attention':
            layer=nn.TransformerEncoderLayer(ch[-1],4,2*ch[-1],dropout=.1,activation='gelu',batch_first=True,norm_first=True)
            self.attention=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            size+=2*ch[-1]
        self.head=nn.Sequential(nn.Linear(size,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,1))
        nn.init.normal_(self.head[-1].weight,std=.01);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        stats=[x.pow(k).mean((2,3)) for k in [1,2,3,4]]
        h=self.stem(x)
        for block,proj in zip(self.stages,self.projections):
            h=block(h);stats.extend([h.mean((2,3)),(h.var((2,3),correction=0)+1e-5).sqrt()])
            z=proj(h).flatten(2);z=z-z.mean(2,keepdim=True)
            stats.append((z@z.transpose(1,2)/z.shape[-1]).flatten(1))
        if self.kind=='attention':
            h=self.attention(h.flatten(2).transpose(1,2));stats.extend([h.mean(1),(h.var(1,correction=0)+1e-5).sqrt()])
        return self.head(torch.cat(stats,1)).squeeze(1)

@torch.no_grad()
def predict(model,pixels,device,batch=16,tta=4):
    model.eval();out=[]
    views=[(0,False),(1,False),(0,True),(1,True)][:tta]
    for start in range(0,len(pixels),batch):
        x=torch.from_numpy(np.array(pixels[start:start+batch],dtype=np.float32)/255.).to(device)
        x=(x-.5)*4;pred=[]
        for rot,flip in views:
            z=torch.rot90(x,rot,(2,3));z=z.flip(3) if flip else z
            pred.append(model(z).cpu().numpy())
        out.append(np.mean(pred,axis=0))
    return np.concatenate(out).astype(float)

def main(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    device=torch.device(args.device)
    if args.device=='mps': assert torch.backends.mps.is_available(),'MPS requires device permission'
    d=np.load(_repo_path(HERE, 'data.npz'));pixels=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r')
    if args.split=='historical':
        tr=d['dev'][d['fold']!=args.fold];va=d['dev'][d['fold']==args.fold]
    else:
        from sklearn.model_selection import KFold
        tr,va=list(KFold(5,shuffle=True,random_state=args.split_seed).split(np.arange(500)))[args.fold]
    assert set(tr).isdisjoint(va)
    folder=_repo_path(HERE, 'runs', args.tag);folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'done.json').exists(),'Completed run protected; use another tag'
    recipe=vars(args)|dict(train=tr.tolist(),validation=va.tolist(),train_ids=d['ids'][tr].tolist(),validation_ids=d['ids'][va].tolist(),
                           code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=str(torch.__version__),numpy=np.__version__)
    if (folder/'recipe.json').exists():
        assert json.loads((folder/'recipe.json').read_text())==recipe
    else:save_json(folder/'recipe.json',recipe)
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    model=Predictor(args.kind,args.width).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.wd)
    target=d['y'][tr];mu=float(target.mean());sd=float(target.std())
    ty=torch.tensor((target-mu)/sd,dtype=torch.float32,device=device)
    # Keep only training pixels on the device; held-out rows are inference batches.
    tx=torch.tensor(np.array(pixels[tr],dtype=np.float32)/255.,device=device);tx=(tx-.5)*4
    avg=None;avg_n=0;curve=[];first_epoch=0;elapsed=0
    checkpoint=folder/'resume.pt'
    if checkpoint.exists():
        ck=torch.load(_repo_io(checkpoint),map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        avg=ck['average'];avg_n=ck['average_n'];curve=ck['curve'];first_epoch=ck['epoch'];elapsed=ck['elapsed']
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        if device.type=='mps':torch.mps.set_rng_state(ck['mps_rng'])
    start=time.monotonic()
    print(json.dumps(dict(start=args.tag,n_train=len(tr),n_validation=len(va),parameters=sum(p.numel() for p in model.parameters()),device=str(device))),flush=True)
    for epoch in range(first_epoch,args.epochs):
        # Five-epoch warm-up followed by a long decay, not an early tiny-LR freeze.
        lr=args.lr*min(1.,(epoch+1)/5)*(.01+.99*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for group in optimizer.param_groups:group['lr']=lr
        model.train();loss_sum=0.;order=rng.permutation(len(tr))
        for offset in range(0,len(tr),args.batch):
            ix=order[offset:offset+args.batch];x=tx[ix]
            rot=int(rng.integers(4));flip=bool(rng.integers(2));x=torch.rot90(x,rot,(2,3));x=x.flip(3) if flip else x
            pred=model(x);loss=F.mse_loss(pred,ty[ix])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            loss_sum+=float(loss.detach().cpu())*len(ix)
        if epoch>=args.epochs-max(10,args.epochs//10):
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};avg_n+=1
            if avg is None:avg=state
            else:
                for k,v in state.items():
                    if v.is_floating_point():avg[k].add_((v-avg[k])/avg_n)
                    else:avg[k]=v
        record=dict(epoch=epoch+1,lr=lr,online_train_rmse=float(np.sqrt(loss_sum/len(tr))*sd),elapsed=elapsed+time.monotonic()-start)
        if epoch==0 or (epoch+1)%args.eval_every==0 or epoch+1==args.epochs:
            vp=predict(model,pixels[va],device,args.batch)*sd+mu
            tp=predict(model,pixels[tr[:64]],device,args.batch,tta=1)*sd+mu
            record.update(validation_rmse=rmse(vp,d['y'][va]),train64_rmse=rmse(tp,d['y'][tr[:64]]))
            np.savez(_repo_io(folder/f'epoch{epoch+1:03d}-predictions.npz'),query=va,prediction=vp)
            print(json.dumps(record),flush=True)
        curve.append(record)
        if (epoch+1)%args.eval_every==0 or epoch+1==args.epochs:
            ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),average=avg,average_n=avg_n,curve=curve,epoch=epoch+1,
                    rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if device.type=='mps' else None,elapsed=record['elapsed'])
            torch.save(ck,_repo_io(folder/'resume-new.pt'));(folder/'resume-new.pt').replace(checkpoint)
            torch.save(dict(state=ck['model'],mu=mu,sd=sd,recipe=recipe),_repo_io(folder/'latest.pt'))
            save_json(folder/'curve.json',curve)
        memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+(torch.mps.driver_allocated_memory() if device.type=='mps' else 0)
        if memory>6*2**30:raise RuntimeError(f'Own process memory budget exceeded: {memory}')
    model.load_state_dict(avg);vp=predict(model,pixels[va],device,args.batch)*sd+mu;tp=predict(model,pixels[tr],device,args.batch)*sd+mu
    torch.save(dict(state=avg,mu=mu,sd=sd,recipe=recipe),_repo_io(folder/'model.pt'))
    np.savez(_repo_io(folder/'predictions.npz'),train=tr,query=va,train_prediction=tp,prediction=vp,ids=d['ids'][va])
    result=dict(tag=args.tag,train_rmse=rmse(tp,d['y'][tr]),validation_rmse=rmse(vp,d['y'][va]),seconds=elapsed+time.monotonic()-start,
                parameters=sum(p.numel() for p in model.parameters()),peak_process_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                mps_driver_bytes=torch.mps.driver_allocated_memory() if device.type=='mps' else 0,averaged_epochs=avg_n)
    if args.split=='historical':
        ix=np.flatnonzero(d['fold']==args.fold);base=d['base083'][ix];y=d['y'][va]
        result.update(base083_rmse=rmse(base,y),fixed_quarter_blend_rmse=rmse(.75*base+.25*vp,y),error_correlation=float(np.corrcoef(vp-y,base-y)[0,1]))
    save_json(folder/'done.json',result);print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--kind',choices=['texture','attention'],default='texture')
    p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026101811);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--wd',type=float,default=.0001);p.add_argument('--width',type=int,default=16)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--device',default='mps');p.add_argument('--eval-every',type=int,default=20)
    p.add_argument('--split',choices=['historical','confirmation'],default='historical');p.add_argument('--split-seed',type=int,default=2026101801)
    main(p.parse_args())
