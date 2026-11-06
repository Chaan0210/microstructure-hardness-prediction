"""Low-dimensional supervised measurement, without per-grain hardness labels."""
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
import hashlib,json,time
import numpy as np
import torch
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

def load():
    b=dict(np.load(_repo_path(ROOT, 'experiments/probabilistic-grains-900/object-bank-train.npz')));d=dict(np.load(_repo_path(HERE, 'data.npz')))
    ids={str(v):i for i,v in enumerate(b['ids'])};ok=list(b['object_keys']);gk=list(b['global_keys']);maxn=int(np.diff(b['offsets']).max())
    z=np.zeros((500,maxn,5));shade=np.zeros((500,maxn));mean=np.zeros((500,maxn));v=np.zeros((500,maxn));area=np.zeros((500,maxn));count=np.zeros((500,maxn));la=np.zeros((500,maxn));tile=np.zeros((500,16,maxn));noise=[];core=[]
    corekeys=['contrast_span','image_mean','image_std','log_area_mean','log_area_std','inv_sqrt_eqd','log_ar_mean','orient_magnitude','orient_c4','orient_s4','het4','pore_area','noise_gray','n_grains','n_effective','border_area']
    for row,ident in enumerate(d['ids']):
        j=ids[str(ident)];a,e=b['offsets'][j:j+2];x=b['objects'][a:e].astype(float);n=len(x);w=b['weights'][a:e].astype(float);g=b['global_X'][j].astype(float)
        span=max(g[gk.index('contrast_span')],10);bright=g[gk.index('ref_bright')]
        q=x[:,[ok.index('intensity_'+s) for s in ['q10','q25','q50','q75','q90']]]
        z[row,:n]=(bright-q)/span;shade[row,:n]=x[:,ok.index('shade_offset')]/span;mean[row,:n]=x[:,ok.index('darkness')]
        v[row,:n]=(x[:,ok.index('intensity_se2')]+x[:,ok.index('shade_se2')])/span**2
        valid=1-x[:,ok.index('pore_flag')];w=w*valid
        if w.sum()==0:w=np.ones(n);valid=np.ones(n)
        area[row,:n]=w/w.sum();count[row,:n]=valid/valid.sum();l=x[:,ok.index('log_area')];la[row,:n]=l-np.sum(l*w/w.sum())
        bins=np.minimum((x[:,ok.index('cx')]*4).astype(int),3)+4*np.minimum((x[:,ok.index('cy')]*4).astype(int),3)
        for k in range(16):tile[row,k,:n]=area[row,:n]*(bins==k)
        noise.append(g[gk.index('noise_gray')]/span)
        c=g[[gk.index(k) for k in corekeys]].copy();c[[13,14]]=np.log1p(c[[13,14]]);core.append(c)
    arrays=dict(z=z,shade=shade,mean=mean,variance=v,area=area,count=count,logarea=la,tile=tile,noise=np.array(noise),core=np.array(core))
    return {k:torch.tensor(a,dtype=torch.float64) for k,a in arrays.items()},d

def features(data,theta):
    # Eleven shared parameters; the three fractions are learned descriptors, not validated phase truth.
    mu=torch.tensor([.15,.30,.50],dtype=torch.float64)+.15*torch.tanh(theta[:3])
    tau=.06*torch.exp(theta[3:6].clamp(-1.5,1.5));slope=.1*torch.tanh(theta[6:9]);blend=torch.sigmoid(theta[9]);vscale=torch.exp(theta[10].clamp(-2,2))
    n=data['noise'];threshold=mu[None,None,:]+n[:,None,None]*slope[None,None,:]
    m=data['mean']+blend*data['shade'];sig=(tau[None,None,:]**2+vscale*data['variance'][:,:,None]+1e-6).sqrt()
    pm=torch.sigmoid((m[:,:,None]-threshold)/sig)
    # Within-object quantiles retain mixed dark/bright material hidden by a merged object's mean.
    z=data['z']+blend*data['shade'][:,:,None]
    pq=torch.sigmoid((z[:,:,:,None]-threshold[:,:,None,:])/(tau[None,None,None,:]**2+.1*vscale*n[:,None,None,None]**2+1e-6).sqrt())
    qw=torch.tensor([.175,.2,.25,.2,.175],dtype=torch.float64);pq=(pq*qw[None,None,:,None]).sum(2)
    w=data['area'][:,:,None];wc=data['count'][:,:,None]
    phi=(pq*w).sum(1);phimean=(pm*w).sum(1);phicount=(pm*wc).sum(1);cov=(pq*w*data['logarea'][:,:,None]).sum(1)
    tw=data['tile'];mass=tw.sum(2);tilephi=torch.einsum('bcn,bnk->bck',tw,pq)/mass.clamp(min=1e-12)[:,:,None]
    het=(((tilephi-phi[:,None])**2)*mass[:,:,None]).sum(1).clamp(min=1e-12).sqrt()
    return torch.cat([data['core'],phi,phimean,phicount,cov,het,phi*n[:,None]],1)

def subset(data,ix):return {k:v[ix] for k,v in data.items()}

def fit(data,y,tr,lam):
    a=subset(data,tr);yy=torch.tensor(y[tr],dtype=torch.float64);ym=yy.mean();ys=yy.std(correction=0);target=(yy-ym)/ys
    theta=torch.zeros(11,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():f0=features(a,theta);fm=f0.mean(0);fs=f0.std(0,correction=0).clamp(min=.01)
    alpha=10.;penalty=torch.eye(f0.shape[1]+1,dtype=torch.float64)*alpha;penalty[0,0]=0
    def solve(th):
        f=(features(a,th)-fm)/fs;X=torch.cat([torch.ones(len(tr),1,dtype=torch.float64),f],1)
        beta=torch.linalg.solve(X.T@X+penalty,X.T@target)
        loss=(X@beta-target).square().mean()+alpha/len(tr)*beta[1:].square().sum()
        return loss,beta
    calls=0
    if lam is not None:
        opt=torch.optim.LBFGS([theta],lr=.5,max_iter=60,max_eval=90,line_search_fn='strong_wolfe',tolerance_grad=1e-7)
        def closure():
            nonlocal calls
            opt.zero_grad();loss,_=solve(theta);loss=loss+lam*theta.square().mean();loss.backward();calls+=1;return loss
        opt.step(closure)
    with torch.no_grad():loss,beta=solve(theta)
    return dict(theta=theta.detach(),feature_mean=fm,feature_std=fs,beta=beta,target_mean=ym,target_std=ys,lambda_=lam,calls=calls,objective=float(loss))

@torch.no_grad()
def predict(model,data,ix):
    f=(features(subset(data,ix),model['theta'])-model['feature_mean'])/model['feature_std']
    X=torch.cat([torch.ones(len(ix),1,dtype=torch.float64),f],1)
    return (model['target_mean']+model['target_std']*(X@model['beta'])).numpy()

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data,d=load();dev=d['dev'];fold=d['fold'];y=d['y'];out=_repo_path(HERE, 'soft-measurement');out.mkdir(exist_ok=True)
    spec=dict(lambdas=[None,.1,1.,10.],ridge_alpha=10.,n_shared_measurement_parameters=11,
        objective='Joint original-image hardness MSE plus coefficient and measurement penalties; exact ridge variable projection.',
        features='Soft mean/quantile darkness fractions, count/area means, spatial heterogeneity, size correlation, noise interactions,16fixed morphology controls.',
        selection='Three inner folds choose regularization, all learned measurement/head/preprocessing re-fit per training subset.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(spec,indent=2));pred=np.empty(400);base=np.empty(400);records=[];start=time.monotonic()
    for k in range(5):
        tr=dev[fold!=k];va=dev[fold==k];candidates=[None,.1,1.,10.];losses=np.zeros(4)
        for it,iv in KFold(3,shuffle=True,random_state=2026101861+k).split(tr):
            for j,lam in enumerate(candidates):
                m=fit(data,y,tr[it],lam);p=predict(m,data,tr[iv]);losses[j]+=((p-y[tr[iv]])**2).sum()
        chosen=candidates[int(np.argmin(losses))];m=fit(data,y,tr,chosen);b=fit(data,y,tr,None)
        pred[fold==k]=predict(m,data,va);base[fold==k]=predict(b,data,va)
        torch.save(dict(model=m,train=tr.tolist(),validation=va.tolist()),_repo_io(out/f'fold{k}.pt'))
        records.append(dict(fold=k,lambda_=chosen,inner_rmse=np.sqrt(losses/len(tr)).tolist(),candidate_rmse=float(np.sqrt(np.mean((pred[fold==k]-y[va])**2))),fixed_measurement_rmse=float(np.sqrt(np.mean((base[fold==k]-y[va])**2))),theta=m['theta'].tolist(),elapsed=time.monotonic()-start))
        print(json.dumps(records[-1]),flush=True);(out/'progress.json').write_text(json.dumps(records,indent=2))
    yy=y[dev];p83=d['base083'];blend=.75*p83+.25*pred;rm=lambda a:float(np.sqrt(np.mean((a-yy)**2)))
    result=dict(folds=records,rmse=rm(pred),fixed_measurement_rmse=rm(base),base083_rmse=rm(p83),quarter_blend_rmse=rm(blend),seconds=time.monotonic()-start)
    np.savez(_repo_io(out/'oof.npz'),ids=d['ids'][dev],y=yy,fold=fold,prediction=pred,control=base,base083=p83,blend=blend)
    (out/'results.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
