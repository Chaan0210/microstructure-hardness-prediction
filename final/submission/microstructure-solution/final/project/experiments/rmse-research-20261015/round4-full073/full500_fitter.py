"""The final residual CNN architecture and its frozen training recipe."""
import gc, resource, time
import numpy as np
import torch
from torch import nn
import training_utils as r
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
    torch.save(checkpoint,output/'model.pt')
    reload_model=ComplementCNN();reload_model.load_state_dict(torch.load(output/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
    rp=predict(reload_model,images,context,va,arm)*ys+(ym if arm=='direct'else 0)
    assert np.array_equal(rp,vp)
    np.savez_compressed(output/'predictions.npz',validation_prediction=vp,train_prediction=tp,validation_index=va,train_index=tr)
    r.save_json(output/'training.json',dict(arm=arm,seed=seed,epochs=epochs,updates=epochs*int(np.ceil(len(tr)/32)),curve=curve,
        train_target_rmse=r.rmse(tp,data['targets']['direct']*ys+ym if arm=='direct'else data['targets']['residual']*ys),
        seconds=time.monotonic()-t0,reload_max_difference=float(np.max(np.abs(rp-vp), initial=0.)),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,meta=data['meta']))
    del model,optimizer,schedule,reload_model,avg;gc.collect()
    return vp

