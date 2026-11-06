"""Train-only supervised grain interactions, with a jointly trainable global branch."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'))
import argparse,copy,hashlib,json,math,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,KFold,train_test_split
from threadpoolctl import threadpool_limits

def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))

def load():
    path=_repo_path(ROOT, 'experiments/probabilistic-grains-900/object-bank-train.npz')
    provenance=json.loads(path.with_name('object-bank-train-provenance.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance['output_sha256']
    b=dict(np.load(_repo_io(path)));d=dict(np.load(_repo_path(HERE, 'data.npz')));index={str(i):j for j,i in enumerate(b['ids'])}
    bags=[];weights=[];summary=[]
    for ident in d['ids']:
        j=index[str(ident)];lo,hi=b['offsets'][j:j+2];x=b['objects'][lo:hi].astype(float);w=b['weights'][lo:hi].astype(float);w/=w.sum()
        # Log redundant positive scale columns so a fallback object is not a huge extrapolation.
        for key in ['area','interior_n','intensity_se2','shade_se2','bbox_max']:
            k=list(b['object_keys']).index(key);x[:,k]=np.log1p(np.maximum(x[:,k],0))
        mu=(x*w[:,None]).sum(0);sd=np.sqrt(((x-mu)**2*w[:,None]).sum(0))
        bags.append(x.astype(np.float32));weights.append(w.astype(np.float32));summary.append(np.r_[mu,sd,b['global_X'][j]])
    assert np.array_equal(b['y'][[index[str(i)] for i in d['ids']]],d['y'])
    return bags,weights,np.array(summary),d,provenance

def context(bags,weights,G,y,tr):
    mx=sum((bags[j]*weights[j][:,None]).sum(0) for j in tr)/len(tr)
    vx=sum(((bags[j]-mx)**2*weights[j][:,None]).sum(0) for j in tr)/len(tr);sx=np.maximum(np.sqrt(vx),1e-5)
    grid=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':[10.,100.,1000.]},
                     cv=KFold(3,shuffle=True,random_state=2026101853),scoring='neg_mean_squared_error',n_jobs=1)
    grid.fit(G[tr],y[tr]);scaler,ridge=grid.best_estimator_.steps[0][1],grid.best_estimator_.steps[1][1]
    ym=float(y[tr].mean());ys=float(y[tr].std())
    return dict(object_mean=mx,object_std=sx,global_mean=scaler.mean_,global_std=scaler.scale_,ym=ym,ys=ys,
                base_w=ridge.coef_/ys,base_b=(ridge.intercept_-ym)/ys,alpha=grid.best_params_['ridge__alpha'])

class Cross(nn.Module):
    def __init__(self):
        super().__init__();self.q=nn.Linear(64,64);self.kv=nn.Linear(64,128);self.out=nn.Linear(64,64)
        self.nq=nn.LayerNorm(64);self.nk=nn.LayerNorm(64);self.nf=nn.LayerNorm(64)
        self.ff=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,64))
    def forward(self,q,k,logweight):
        b,m,_=q.shape;n=k.shape[1]
        qq=self.q(self.nq(q)).reshape(b,m,4,16).transpose(1,2)
        kk,vv=self.kv(self.nk(k)).reshape(b,n,2,4,16).permute(2,0,3,1,4).unbind(0)
        z=F.scaled_dot_product_attention(qq,kk,vv,attn_mask=logweight[:,None,None,:],dropout_p=0.)
        q=q+self.out(z.transpose(1,2).reshape(b,m,64));return q+self.ff(self.nf(q))

class Model(nn.Module):
    def __init__(self,od,gd,ctx,kind):
        super().__init__();self.kind=kind
        self.phi=nn.Sequential(nn.Linear(od+gd,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.gproj=nn.Linear(gd,64);self.inducing=nn.Parameter(torch.randn(1,32,64)*.02)
        self.to_latent=nn.ModuleList([Cross(),Cross()]);self.to_objects=nn.ModuleList([Cross(),Cross()])
        self.base=nn.Linear(gd,1);self.head=nn.Sequential(nn.Linear(4*64+gd,128),nn.SiLU(),nn.Dropout(.2),nn.Linear(128,1))
        with torch.no_grad():self.base.weight.copy_(torch.tensor(ctx['base_w'][None],dtype=torch.float32));self.base.bias.fill_(ctx['base_b'])
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,w,g,mask):
        h=self.phi(torch.cat([x,g[:,None].expand(-1,x.shape[1],-1)],2))
        if self.kind=='attention':
            z=self.inducing.expand(len(x),-1,-1)+self.gproj(g)[:,None]
            lw=torch.where(mask,torch.log(w.clamp(min=1e-12)),torch.full_like(w,-1e9));uniform=torch.zeros(len(x),32)
            for a,b in zip(self.to_latent,self.to_objects):z=a(z,h,lw);h=b(h,z,uniform)
        means=[]
        for ww in [w,mask.float()/mask.sum(1,keepdim=True)]:
            mu=(h*ww[:,:,None]).sum(1);sd=(((h-mu[:,None])**2*ww[:,:,None]).sum(1)+1e-6).sqrt();means.extend([mu,sd])
        return self.base(g)[:,0]+self.head(torch.cat(means+[g],1))[:,0]

class Data:
    def __init__(self,bags,weights,G,ctx):
        self.bags=[torch.from_numpy(np.clip((x-ctx['object_mean'])/ctx['object_std'],-10,10).astype(np.float32)) for x in bags]
        self.weights=[torch.from_numpy(w) for w in weights]
        self.g=torch.tensor(np.clip((G-ctx['global_mean'])/ctx['global_std'],-10,10),dtype=torch.float32)
    def batch(self,ix):
        n=max(len(self.bags[j]) for j in ix);x=torch.zeros(len(ix),n,self.bags[0].shape[1]);w=torch.zeros(len(ix),n);mask=torch.zeros(len(ix),n,dtype=torch.bool)
        for row,j in enumerate(ix):
            nn_=len(self.bags[j]);x[row,:nn_]=self.bags[j];w[row,:nn_]=self.weights[j];mask[row,:nn_]=True
        return x,w,self.g[ix],mask

@torch.no_grad()
def predict(model,data,ix,ctx):
    model.eval();out=[]
    for first in range(0,len(ix),16):out.extend(model(*data.batch(ix[first:first+16])).numpy()*ctx['ys']+ctx['ym'])
    return np.array(out)

def fit(bags,weights,G,y,tr,q,kind,seed,epochs,folder,select=False):
    folder.mkdir(parents=True,exist_ok=True);ctx=context(bags,weights,G,y,tr);data=Data(bags,weights,G,ctx)
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Model(bags[0].shape[1],G.shape[1],ctx,kind)
    initial_base=model.base.weight.detach().clone();initial_bias=model.base.bias.detach().clone()
    base_ids={id(p) for p in model.base.parameters()};opt=torch.optim.AdamW([
        dict(params=model.base.parameters(),lr=.0001),dict(params=[p for p in model.parameters() if id(p) not in base_ids],lr=.001)],weight_decay=.01)
    target=torch.tensor((y-ctx['ym'])/ctx['ys'],dtype=torch.float32)
    basepred=predict(model,data,q,ctx);best_mse=float(np.mean((basepred-y[q])**2));best_epoch=0;best=copy.deepcopy(model.state_dict());curve=[];start=time.monotonic()
    for epoch in range(epochs):
        factor=.05+.95*.5*(1+math.cos(math.pi*epoch/epochs));opt.param_groups[0]['lr']=.0001*factor;opt.param_groups[1]['lr']=.001*factor
        model.train();order=rng.permutation(tr);loss_sum=0
        for first in range(0,len(order),16):
            ix=order[first:first+16];p=model(*data.batch(ix));mse=F.mse_loss(p,target[ix])
            penalty=.01*((model.base.weight-initial_base).square().sum()+(model.base.bias-initial_bias).square().sum())
            loss=mse+penalty;assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step();loss_sum+=float(mse.detach())*len(ix)
        rec=dict(epoch=epoch+1,online_train_rmse=float(np.sqrt(loss_sum/len(tr))*ctx['ys']),seconds=time.monotonic()-start)
        if (epoch+1)%10==0 or epoch+1==epochs:
            pr=predict(model,data,q,ctx);score=float(np.mean((pr-y[q])**2));rec['validation_rmse']=float(np.sqrt(score))
            if select and score<best_mse:best_mse=score;best_epoch=epoch+1;best=copy.deepcopy(model.state_dict())
            print(folder.name,kind,json.dumps(rec),flush=True);save(folder/'curve.json',curve+[rec])
        curve.append(rec)
    if select:model.load_state_dict(best)
    pr=predict(model,data,q,ctx);tp=predict(model,data,tr,ctx)
    torch.save(dict(state=model.state_dict(),context=ctx,train=tr.tolist(),query=q.tolist(),kind=kind,seed=seed,epochs=best_epoch if select else epochs),_repo_io(folder/'model.pt'))
    np.savez(_repo_io(folder/'predictions.npz'),train=tr,query=q,prediction=pr,train_prediction=tp,base_prediction=basepred)
    result=dict(selected_epoch=best_epoch if select else epochs,rmse=rmse(pr,y[q]),base_rmse=rmse(basepred,y[q]),train_rmse=rmse(tp,y[tr]),seconds=time.monotonic()-start,alpha=ctx['alpha'])
    save(folder/'done.json',result);return result

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    bags,weights,G,d,prov=load();tr=d['dev'][d['fold']!=a.fold];va=d['dev'][d['fold']==a.fold]
    A,B=train_test_split(tr,test_size=.2,random_state=2026101847+a.fold)
    out=_repo_path(HERE, 'runs', a.tag);out.mkdir(parents=True,exist_ok=True)
    save(out/'recipe.json',vars(a)|dict(A=A.tolist(),B=B.tolist(),T=tr.tolist(),V=va.tolist(),object_cache_provenance=prov,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        target='Direct original-image hardness; no fixed residual targets or base predictions supplied as input. Ridge initializes a jointly optimized global linear branch.',
        transforms='None; preserve signed orientation. Grain order is a set, all objects retained.'))
    r=fit(bags,weights,G,d['y'],A,B,a.kind,a.seed,160,out/'selector',select=True)
    r2=fit(bags,weights,G,d['y'],tr,va,a.kind,a.seed,r['selected_epoch'],out/'refit')
    p=np.load(_repo_io(out/'refit/predictions.npz'))['prediction'];base=d['base083'][d['fold']==a.fold]
    save(out/'done.json',dict(selector=r,refit=r2,base083_rmse=rmse(base,d['y'][va]),quarter_blend_rmse=rmse(.75*base+.25*p,d['y'][va])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--kind',choices=['attention','deepset'],default='attention');p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026101841)
    with threadpool_limits(limits=2):main(p.parse_args())
