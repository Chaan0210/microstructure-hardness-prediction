# Full500 variant: only empty-query reload diagnostics change.
# Derived by derive_fitter.py from the immutable round3 source.
# Only training-size assertion and non-operative checkpoint shrinkage metadata change.
# The experiment runner, not this fitter, selects ensemble weights internally.
"""CPU-only matched CNN objective/conditioning study with honest tabular targets."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import copy
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'2')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import research as r
import numpy as np
from PIL import Image
import torch
from torch import nn
from threadpoolctl import threadpool_limits

ROUND2=_repo_path(HERE.parent, 'round2')
ARMS=('direct','residual','context','aux','context_only')
SEEDS=(2026101501,2026101502,2026101503)
EPOCHS=40
SHRINK=.25


class ComplementCNN(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList();inc=1
        for i,out in enumerate((16,32,48,64)):
            self.blocks.append(nn.Sequential(nn.Conv2d(inc,out,7 if i==0 else 3,stride=4 if i==0 else 2,padding=3 if i==0 else 1),
                                            nn.GroupNorm(4,out),nn.SiLU()));inc=out
        self.main=nn.Sequential(nn.Linear(328,64),nn.SiLU(),nn.Dropout(.25),nn.Linear(64,1))
        self.aux=nn.Linear(322,1)
        nn.init.zeros_(self.main[-1].weight);nn.init.zeros_(self.main[-1].bias)
        nn.init.zeros_(self.aux.weight);nn.init.zeros_(self.aux.bias)

    def forward(self,x,context,arm):
        if arm=='context_only':image=torch.zeros((len(context),322),device=context.device,dtype=context.dtype)
        else:
            stats=[x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()];h=x
            for block in self.blocks:
                h=block(h);stats.extend([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt()])
            image=torch.cat(stats,1)
        ctx=context if arm in ('context','aux','context_only')else torch.zeros_like(context)
        return self.main(torch.cat([image,ctx],1)).squeeze(1),self.aux(image).squeeze(1)


def load_data():
    d=dict(np.load(_repo_path(HERE.parent, 'inputs.npz')))
    d['images']=np.stack([np.asarray(Image.open(_repo_path(r.ROOT, f'data/train/{i}.png')),np.float32)/255 for i in d['ids']])[:,None]
    assert d['images'].shape==(500,1,256,256)
    return d


def fold_data(d,partition,fold):
    folder=_repo_path(ROUND2, f'full500-{SEEDS[partition]}');outer=np.load(_repo_io(folder/f'fold{fold}.npz'))
    ids=d['ids'];pos={str(i):j for j,i in enumerate(ids)}
    tr=np.array([pos[str(i)]for i in outer['train_ids']]);va=outer['validation_index']
    assert not set(tr)&set(va)
    base=np.zeros(500);disagreement=np.zeros(500);coverage=np.zeros(500,int);sources=[]
    for j in range(3):
        z=np.load(_repo_io(folder/f'fold{fold}-inner{j}.npz'))
        it={pos[str(i)]for i in z['train_ids']};iv=np.array([pos[str(i)]for i in z['validation_ids']])
        assert not it&set(iv)and (it|set(iv))==set(tr)and not(it|set(iv))&set(va)
        base[iv]=z['base'];disagreement[iv]=z['tab']-z['ridge'];coverage[iv]+=1
        sources.append(dict(train_ids=z['train_ids'].tolist(),validation_ids=z['validation_ids'].tolist(),sha256=r.digest(folder/f'fold{fold}-inner{j}.npz')))
    assert np.all(coverage[tr]==1)and np.all(coverage[va]==0)
    base[va]=outer['base'];disagreement[va]=outer['tab']-outer['ridge']
    key=list(d['keys']);raw=np.c_[base,disagreement,d['noise'],d['X'][:,key.index('u_ph')],
        np.log1p(np.maximum(np.nan_to_num(d['X'][:,key.index('n_obj')],nan=0),0)),d['X'][:,key.index('coverage')]]
    med=np.nanmedian(raw[tr],axis=0);raw=np.where(np.isfinite(raw),raw,med)
    cm=raw[tr].mean(0);cs=np.maximum(raw[tr].std(0),1e-6);context=((raw-cm)/cs).astype(np.float32)
    xm=float(d['images'][tr].mean());xs=float(d['images'][tr].std());ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std())
    images=((d['images']-xm)/xs).astype(np.float32)
    targets=dict(direct=((d['y'][tr]-ym)/ys).astype(np.float32),residual=((d['y'][tr]-base[tr])/ys).astype(np.float32))
    failed=np.nan_to_num(d['X'][:,key.index('n_obj')],nan=0)<20
    meta=dict(partition=partition,fold=fold,train_ids=ids[tr].tolist(),validation_ids=ids[va].tolist(),
        image_mean=xm,image_std=xs,target_mean=ym,target_std=ys,context_mean=cm.tolist(),context_std=cs.tolist(),context_median=med.tolist(),
        inner_sources=sources,outer_tabular_sha256=r.digest(folder/f'fold{fold}.npz'))
    return dict(images=images,context=context,targets=targets,train=tr,valid=va,base=base,failed=failed,meta=meta)


@torch.no_grad()
def predict(model,images,context,indices,arm):
    model.eval();out=[]
    for first in range(0,len(indices),32):
        ix=indices[first:first+32];xb=images[ix];ctx=torch.from_numpy(context[ix]);views=[]
        transforms=((0,False),(1,False),(0,True),(1,True))
        for rot,flip in transforms:
            xx=np.rot90(xb,rot,axes=(2,3));xx=xx[:,:,:,::-1]if flip else xx
            pred,_=model(torch.from_numpy(np.ascontiguousarray(xx)),ctx,arm);views.append(pred.numpy())
        out.extend(np.mean(views,axis=0))
    return np.array(out,dtype=float)


def fit_one(data,arm,seed,output,epochs=EPOCHS):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=ComplementCNN();optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.0001)
    tr=data['train'];va=data['valid'];target=data['targets']['direct'if arm=='direct'else'residual']
    direct=data['targets']['direct'];images=data['images'];context=data['context'];ys=data['meta']['target_std'];ym=data['meta']['target_mean']
    assert len(target)==len(tr) and len(tr)>0
    avg=None;curve=[];t0=time.monotonic()
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(tr));loss_sum=0.;aux_sum=0.
        for first in range(0,len(tr),32):
            ix=order[first:first+32];ii=tr[ix];rot=int(rng.integers(4));flip=bool(rng.random()<.5)
            x=np.rot90(images[ii],rot,axes=(2,3));x=x[:,:,:,::-1]if flip else x
            pred,aux=model(torch.from_numpy(np.ascontiguousarray(x)),torch.from_numpy(context[ii]),arm)
            main_loss=nn.functional.mse_loss(pred,torch.from_numpy(target[ix]));aux_loss=nn.functional.mse_loss(aux,torch.from_numpy(direct[ix]))
            loss=main_loss+.2*aux_loss if arm=='aux'else main_loss
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            assert torch.isfinite(norm)
            loss_sum+=float(main_loss.detach())*len(ix);aux_sum+=float(aux_loss.detach())*len(ix)
        schedule.step();curve.append(dict(epoch=epoch+1,augmented_train_rmse=np.sqrt(loss_sum/len(tr))*ys,auxiliary_rmse=np.sqrt(aux_sum/len(tr))*ys))
        if epoch>=epochs-5:
            state={k:v.detach().clone()for k,v in model.state_dict().items()}
            if avg is None:avg={k:v/5 for k,v in state.items()}
            else:
                for k,v in state.items():avg[k]+=v/5
        if(epoch+1)%20==0:print(arm,'epoch',epoch+1,'train',round(curve[-1]['augmented_train_rmse'],3),'seconds',round(time.monotonic()-t0,1),flush=True)
    model.load_state_dict(avg)
    vp=predict(model,images,context,va,arm)*ys
    tp=predict(model,images,context,tr,arm)*ys
    if arm=='direct':vp+=ym;tp+=ym
    output.mkdir(exist_ok=False)
    checkpoint=dict(state_dict=avg,normalization=data['meta'],arm=arm,seed=seed,epochs=epochs,architecture='native256-multiscale-moment-CNN',
        context_dimensions=6,shrinkage=None,torch_version=str(torch.__version__))
    torch.save(checkpoint,_repo_io(output/'model.pt'))
    reload_model=ComplementCNN();reload_model.load_state_dict(torch.load(_repo_io(output/'model.pt'),map_location='cpu',weights_only=True)['state_dict'])
    rp=predict(reload_model,images,context,va,arm)*ys+(ym if arm=='direct'else 0)
    assert np.array_equal(rp,vp)
    np.savez_compressed(_repo_io(output/'predictions.npz'),validation_prediction=vp,train_prediction=tp,validation_index=va,train_index=tr)
    r.save_json(output/'training.json',dict(arm=arm,seed=seed,epochs=epochs,updates=epochs*int(np.ceil(len(tr)/32)),curve=curve,
        train_target_rmse=r.rmse(tp,data['targets']['direct']*ys+ym if arm=='direct'else data['targets']['residual']*ys),
        seconds=time.monotonic()-t0,reload_max_difference=float(np.max(np.abs(rp-vp), initial=0.)),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,meta=data['meta']))
    del model,optimizer,schedule,reload_model,avg;gc.collect()
    return vp

