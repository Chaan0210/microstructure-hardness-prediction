"""Direct supervised adaptation of eligible MicroNet/ImageNet ResNet50 encoders."""
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
import argparse,hashlib,json,math,resource,time
import numpy as np
import torch,torchvision
from torch import nn
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,KFold
from threadpoolctl import threadpool_limits

def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))

class RMSNorm(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.mean_=X.mean(0);self.scale_=np.maximum.reduce([X.std(0),.1*np.sqrt(np.mean(X*X,0)),np.full(X.shape[1],1e-4)]);return self
    def transform(self,X):return np.clip((X-self.mean_)/self.scale_,-8,8)

def backbone(arm):
    record=json.loads((_repo_path(ROOT, 'experiments/micronet-frozen-205/plan.json')).read_text())['weights'][arm]
    path=(_repo_path(ROOT, record['path'])).resolve();assert path.is_relative_to(ROOT)
    m=torchvision.models.resnet50(weights=None);m.fc=nn.Identity();state=torch.load(_repo_io(path),map_location='cpu',weights_only=True)
    if arm=='imagenet':state={k:v for k,v in state.items() if not k.startswith('fc.')}
    m.load_state_dict(state,strict=True);return m,record

class Regressor(nn.Module):
    def __init__(self,body,norm,ridge,ym,ys):
        super().__init__();self.body=body
        self.register_buffer('mean',torch.tensor(norm.mean_,dtype=torch.float32));self.register_buffer('scale',torch.tensor(norm.scale_,dtype=torch.float32))
        self.register_buffer('rgbmean',torch.tensor([.485,.456,.406],dtype=torch.float32)[None,:,None,None]);self.register_buffer('rgbstd',torch.tensor([.229,.224,.225],dtype=torch.float32)[None,:,None,None])
        self.head=nn.Linear(len(norm.mean_),1)
        with torch.no_grad():self.head.weight.copy_(torch.tensor(ridge.coef_[None]/ys,dtype=torch.float32));self.head.bias.fill_((ridge.intercept_-ym)/ys)
    def forward(self,gray):
        x=(gray.expand(-1,3,-1,-1)-self.rgbmean)/self.rgbstd
        f=self.body(x);raw=torch.stack([gray.pow(k).mean((1,2,3)) for k in [1,2,3,4]],1)
        f=torch.cat([f,raw],1);z=((f-self.mean)/self.scale).clamp(-8,8);return self.head(z)[:,0]

@torch.no_grad()
def predict(model,px,indices,ym,ys,batch=8,views=4):
    model.eval();out=[]
    for first in range(0,len(indices),batch):
        x=torch.tensor(np.array(px[indices[first:first+batch]],dtype=np.float32)/255.,device='mps');p=[]
        for rot in range(views):p.append(model(torch.rot90(x,rot,(2,3))).cpu().numpy())
        out.extend(np.mean(p,0)*ys+ym)
    return np.array(out)

def main(a):
    assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.35)
    d=np.load(_repo_path(HERE, 'data.npz'));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');tr=d['dev'][d['fold']!=a.fold];q=d['dev'][d['fold']==a.fold]
    old=np.load(_repo_path(ROOT, 'experiments/micronet-frozen-205/features.npz'));lookup={str(v):i for i,v in enumerate(old['ids'])}
    assert set(d['ids'][np.r_[tr,q]])<=set(lookup);raw=[]
    for j in d['dev']:
        im=px[j].astype(float)/255.;raw.append([float(np.mean(im**k)) for k in [1,2,3,4]])
    F=np.c_[old[a.arm][[lookup[str(v)] for v in d['ids'][d['dev']]]],np.array(raw)];pos={int(j):i for i,j in enumerate(d['dev'])};ti=np.array([pos[int(j)] for j in tr]);qi=np.array([pos[int(j)] for j in q])
    search=GridSearchCV(make_pipeline(RMSNorm(),Ridge()),{'ridge__alpha':[10.,100.,1000.,10000.]},cv=KFold(3,shuffle=True,random_state=2026101903+a.fold),scoring='neg_mean_squared_error',n_jobs=1)
    search.fit(F[ti],d['y'][tr]);norm,ridge=[v for _,v in search.best_estimator_.steps];alpha=search.best_params_['ridge__alpha']
    ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());torch.manual_seed(a.seed);torch.mps.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    body,provenance=backbone(a.arm);model=Regressor(body,norm,ridge,ym,ys).to('mps');out=_repo_path(HERE, 'runs', a.tag);out.mkdir(parents=True,exist_ok=True);assert not (out/'done.json').exists()
    recipe=vars(a)|dict(train=tr.tolist(),query=q.tolist(),train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist(),head_alpha=alpha,
        weight=provenance,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        image_input='Native256grayscale replicatedRGB,officialImageNetnormalization,C4rotationsonly,noresizing/crop/photometricaugmentation',
        representation='EntireResNet50fine-tuned;frozenBNrunningstats;linearRidge-warmstartedhead onmean2048+raw4moments',
        objective='DirecthardnessMSE +sameRidgeheadpenalty +pretrainedencoderparameteranchor',
        scope='Newmodelend-to-endexcludesoutervalidationimages;externalpretrainedweightseligible.Matchedfull083comparisonconditionalonitslegacyrepresentations.')
    save(out/'recipe.json',recipe)
    # The reused mean cache corresponds to original-view inference exactly within prior device tolerances.
    original=predict(model,px,q,ym,ys,a.batch,views=1);cached=search.predict(F[qi]);cache_error=float(abs(original-cached).max());assert cache_error<.01
    initial=predict(model,px,q,ym,ys,a.batch);print('INITIAL',a.arm,json.dumps(dict(frozen_rmse=rmse(initial,d['y'][q]),alpha=alpha,cache_max_difference=cache_error)),flush=True)
    np.savez(_repo_io(out/'frozen-predictions.npz'),train=tr,query=q,prediction=initial)
    prior=[v.detach().clone() for v in model.body.parameters()]
    optimizer=torch.optim.AdamW([dict(params=model.body.parameters(),lr=a.body_lr),dict(params=model.head.parameters(),lr=a.head_lr)],weight_decay=0.)
    tx=torch.tensor(np.array(px[tr],dtype=np.float32)/255.,device='mps');ty=torch.tensor((d['y'][tr]-ym)/ys,dtype=torch.float32,device='mps')
    start=time.monotonic();curve=[];avg=None;navg=0;first_epoch=0;elapsed=0
    resume=out/'resume.pt'
    if resume.exists():
        r=torch.load(_repo_io(resume),map_location='cpu',weights_only=False);assert r['recipe']==recipe;model.load_state_dict(r['state']);optimizer.load_state_dict(r['optimizer']);first_epoch=r['epoch'];curve=r['curve'];avg=r['average'];navg=r['navg'];elapsed=r['elapsed'];rng.bit_generator.state=r['rng'];torch.set_rng_state(r['torch_rng']);torch.mps.set_rng_state(r['mps_rng'])
        for s in optimizer.state.values():
            for k,v in s.items():
                if torch.is_tensor(v) and k!='step':s[k]=v.to('mps')
    for epoch in range(first_epoch,a.epochs):
        factor=min(1.,(epoch+1)/3)*(.02+.98*.5*(1+math.cos(math.pi*epoch/a.epochs)))
        optimizer.param_groups[0]['lr']=a.body_lr*factor;optimizer.param_groups[1]['lr']=a.head_lr*factor
        model.train()
        for layer in model.body.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        order=rng.permutation(len(tr));total=0.
        for first in range(0,len(tr),a.batch):
            ix=order[first:first+a.batch];rot=int(rng.integers(4));p=model(torch.rot90(tx[ix],rot,(2,3)));mse=(p-ty[ix]).square().mean()
            reg=alpha/len(tr)*model.head.weight.square().sum()+a.anchor*sum((v-old).square().sum() for v,old in zip(model.body.parameters(),prior))
            loss=mse+reg;assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(mse.detach().cpu())*len(ix)
            if (first//a.batch+1)%20==0:torch.mps.synchronize();torch.mps.empty_cache()
        if epoch>=a.epochs-5:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};navg+=1
            if avg is None:avg=state
            else:
                for k,v in state.items():
                    if v.is_floating_point():avg[k].add_((v-avg[k])/navg)
                    else:avg[k]=v
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+torch.mps.driver_allocated_memory();assert mem<6*2**30
        rec=dict(epoch=epoch+1,online_training_rmse=float(np.sqrt(total/len(tr))*ys),seconds=elapsed+time.monotonic()-start,memory=mem)
        if epoch==0 or (epoch+1)%10==0 or epoch+1==a.epochs:
            pred=predict(model,px,q,ym,ys,a.batch);rec['validation_rmse']=rmse(pred,d['y'][q]);np.savez(_repo_io(out/f'epoch{epoch+1:03d}.npz'),query=q,prediction=pred);print(json.dumps(rec),flush=True)
            r=dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),recipe=recipe,epoch=epoch+1,curve=curve+[rec],average=avg,navg=navg,elapsed=rec['seconds'],rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),ym=ym,ys=ys)
            torch.save(r,_repo_io(out/'resume-new.pt'));(out/'resume-new.pt').replace(resume)
        curve.append(rec);save(out/'curve.json',curve)
    model.load_state_dict(avg);pred=predict(model,px,q,ym,ys,a.batch);tp=predict(model,px,tr,ym,ys,a.batch)
    torch.save(dict(state=avg,recipe=recipe,ym=ym,ys=ys),_repo_io(out/'model.pt'));np.savez(_repo_io(out/'predictions.npz'),train=tr,query=q,prediction=pred,train_prediction=tp)
    ref=np.load(_repo_path(ROOT, 'experiments/rmse-research-20261015/inputs.npz'));assert np.array_equal(ref['dev'],d['dev']);ix=d['fold']==a.fold;base=d['base083'][ix];p21=ref['C73_cnn'][0,ix];eligible=~d['failed'][ix]
    result=dict(train_rmse=rmse(tp,d['y'][tr]),rmse=rmse(pred,d['y'][q]),frozen_rmse=rmse(initial,d['y'][q]),base083_rmse=rmse(base,d['y'][q]),G021_rmse=rmse(p21,d['y'][q]),quarter_blend_rmse=rmse(base+.25*eligible*(pred-base),d['y'][q]),replace_image_branch_rmse=rmse(base+.16*eligible*(pred-p21),d['y'][q]),seconds=elapsed+time.monotonic()-start)
    save(out/'done.json',result);print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--arm',choices=['micronet','imagenet'],default='micronet');p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026101901)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--batch',type=int,default=8);p.add_argument('--body-lr',type=float,default=.00003);p.add_argument('--head-lr',type=float,default=.0001);p.add_argument('--anchor',type=float,default=.0001)
    with threadpool_limits(limits=2):main(p.parse_args())
