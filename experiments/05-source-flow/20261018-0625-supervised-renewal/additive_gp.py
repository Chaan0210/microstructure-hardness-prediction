"""Joint linear + additive + pairwise-interaction GP on physical measurements.

Unlike a full-vector RBF/distribution kernel, the group ANOVA kernel expresses
smooth lower-order physical relationships without requiring all104 coordinates
to be close. All preprocessing and MAP hyperparameter fitting are train-only.
"""
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
import argparse,hashlib,json,time
import numpy as np
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

def groups(keys):
    out={}
    for j,key in enumerate(keys):
        if '_x_' in key:g='explicit_interactions'
        elif key.startswith('vstd_'):g='view_disagreement'
        elif 'pore' in key:g='pores'
        elif any(s in key for s in ['sizefield','tile','het4','_f16','_f32','m_la_std','sp_field','neighbors','sp_contiguity']):g='spatial_heterogeneity'
        elif any(s in key for s in ['ar_','orient','coherence','stcoh','m_el','ang_','solidity']) or key=='sp_ar':g='shape_orientation'
        elif any(s in key for s in ['u_ph','m_ph','sp_','z_f','p34']):g='phase'
        elif any(s in key for s in ['eqd','sqrt_d','area','logA','n_obj','obj_per','density','m_la_mean']):g='size'
        else:g='photometry_quality'
        out.setdefault(g,[]).append(j)
    return out

def prepare(X,train):
    median=np.nan_to_num(np.nanmedian(X[train],0));Z=np.where(np.isfinite(X),X,median);mu=Z[train].mean(0);sd=np.maximum(Z[train].std(0),.001);return np.clip((Z-mu)/sd,-6,6),dict(median=median,mean=mu,scale=sd)

def distances(A,B,groups):return np.stack([np.maximum(((A[:,None,c]-B[None,:,c])**2).mean(2),0) for c in groups.values()])

def kernel(theta,D,L,ng,mode):
    nc=3 if mode=='anova' else 2;amps=theta[:nc].exp();noise=theta[nc].exp();ell=theta[nc+1:nc+1+ng].exp();weights=theta[nc+1+ng:].softmax(0)
    K=torch.exp(-.5*D/ell[:,None,None].square());W=weights[:,None,None]*K;s=W.sum(0);out=amps[0]*L+amps[1]*s
    if mode=='anova':
        denom=(1-weights.square().sum()).clamp(min=1e-6);pair=(s.square()-W.square().sum(0))/denom;out=out+amps[2]*pair
    return out,noise

def fit(X,y,tr,query,G,mode):
    Z,norm=prepare(X,tr);A=Z[tr];Q=Z[query];DD=torch.tensor(distances(A,A,G),dtype=torch.float64);DQ=torch.tensor(distances(Q,A,G),dtype=torch.float64);L=torch.tensor(A@A.T/A.shape[1]);Lq=torch.tensor(Q@A.T/A.shape[1]);ys=float(y[tr].std());ym=float(y[tr].mean());target=torch.tensor((y[tr]-ym)/ys,dtype=torch.float64);n=len(tr);ng=len(G);nc=3 if mode=='anova' else 2
    prior=np.r_[np.array([0.,-1.,-1.])[:nc],np.log(.25),np.full(ng,np.log(2.)),np.zeros(ng)];std=np.r_[np.full(nc,1.5),1.,np.ones(ng),np.ones(ng)];bounds=[(-6,6)]*nc+[(-5.3,1.4)]+[(-2.,4.)]*ng+[(-3,3)]*ng;best=None;logs=[]
    def objective(v):
        theta=torch.tensor(v,dtype=torch.float64,requires_grad=True);K,noise=kernel(theta,DD,L,ng,mode);chol=torch.linalg.cholesky(K+(noise+1e-7)*torch.eye(n,dtype=torch.float64));dual=torch.cholesky_solve(target[:,None],chol)[:,0]
        loss=.5*(target@dual)+chol.diagonal().log().sum()+.5*n*np.log(2*np.pi)+.5*(((theta-torch.tensor(prior))/torch.tensor(std))**2).sum();loss.backward();return float(loss.detach()),theta.grad.numpy().copy()
    start=time.monotonic()
    for initial_length in [1.,4.]:
        init=prior.copy();init[nc+1:nc+1+ng]=np.log(initial_length);r=minimize(objective,init,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=200,ftol=1e-10,gtol=1e-5));logs.append(dict(initial_length=initial_length,objective=float(r.fun),iterations=int(r.nit),success=bool(r.success),message=str(r.message),gradient_norm=float(np.linalg.norm(r.jac))))
        if best is None or r.fun<best.fun:best=r
    with torch.no_grad():
        th=torch.tensor(best.x,dtype=torch.float64);K,noise=kernel(th,DD,L,ng,mode);cross,_=kernel(th,DQ,Lq,ng,mode);chol=torch.linalg.cholesky(K+(noise+1e-7)*torch.eye(n));dual=torch.cholesky_solve(target[:,None],chol)[:,0];p=(ym+ys*(cross@dual)).numpy();tp=(ym+ys*(K@dual)).numpy()
    return p,dict(theta=best.x,dual=dual.numpy(),ym=ym,ys=ys,train_features=A,**norm),dict(optimization=logs,seconds=time.monotonic()-start,training_fitted_rmse=float(np.mean((tp-y[tr])**2)**.5),fitted_noise_sd=ys*float(noise.sqrt()),noise_interpretation='Model residual parameter, not evidence of irreducible label noise.')

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));ids=d['ids'][d['dev']];y=d['y'][d['dev']];fold=d['fold'];fp=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(fp));pos={str(v):i for i,v in enumerate(z['ids'])};X=z['X'][[pos[str(v)] for v in ids]].astype(float);keys=z['keys'].tolist();G=groups(keys);out=_repo_path(HERE, 'additive-gp');out.mkdir(exist_ok=True)
    recipe=dict(groups={k:[keys[i] for i in v] for k,v in G.items()},covariance='Joint linear plus weighted sum of8semantic-group RBF kernels plus weightedpairwiseproducts. GlobalRBF+linear control. Positiveamplitudes/noise, lognormalMAPpriors, two fixed lengthscaleinitializations chosenbytrainingposterior.',
        normalization='Trainingmedian/mean/std,clip±6; no query fitting.',selection='No outer-grid. Fixedarchitecture/priors; MAP fitting solelyonT320. All5originalfolds.',primary='Fixed25%newGP/75%complete083, existingfailed-maskfallback; standalonealso reported.',
        feature_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Conditionalhistoricalphysicalfeatures; newpredictionlearningisoutertrain-only.')
    (out/'recipe.json').write_text(json.dumps(recipe,indent=2));pred={k:np.zeros(400) for k in ['global','anova']};rows=[]
    for f in range(5):
        tr=np.flatnonzero(fold!=f);q=np.flatnonzero(fold==f)
        for mode,gg in [('global',{'all':list(range(X.shape[1]))}),('anova',G)]:
            file=out/f'{mode}-f{f}.npz';report=out/f'{mode}-f{f}.json'
            if file.exists():r=np.load(_repo_io(file));p=r['prediction'];details=json.loads(report.read_text())
            else:
                p,state,details=fit(X,y,tr,q,gg,mode);np.savez(_repo_io(out/f'{mode}-f{f}-model.npz'),train=tr,**state);np.savez(_repo_io(file),train=tr,query=q,prediction=p);details.update(fold=f,mode=mode,rmse=float(np.mean((p-y[q])**2)**.5));report.write_text(json.dumps(details,indent=2));print(json.dumps(details),flush=True)
            pred[mode][q]=p;rows.append(details)
        if a.probe:break
    if a.probe:return
    for mode in ['global','anova']:pred[mode+'_system']=d['base083']+.25*(~d['failed'])*(pred[mode]-d['base083'])
    pred['base083']=d['base083'];r={k:dict(rmse=float(np.mean((v-y)**2)**.5),fold_rmse=[float(np.mean((v[fold==f]-y[fold==f])**2)**.5) for f in range(5)]) for k,v in pred.items()};np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**pred);(out/'results.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true')
    with threadpool_limits(2):main(ap.parse_args())
