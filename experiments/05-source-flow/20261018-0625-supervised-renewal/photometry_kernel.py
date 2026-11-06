"""Source-only balanced geometry/photometry kernels with honest head selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import numpy as np,json,time
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

def norm(X):
    mean=X.mean(0);scale=np.maximum.reduce([X.std(0),.1*np.sqrt(np.mean(X*X,0)),np.full(X.shape[1],1e-4)])
    return np.clip((X-mean)/scale,-8,8),mean,scale
def kernel(X,Q):
    linear=Q@X.T;dist=np.maximum((Q*Q).sum(1)[:,None]+(X*X).sum(1)[None]-2*linear,0)
    return .5*linear+.5*np.exp(-.5*dist)
def fit(X,photo,y,tr,alpha):
    P,pm,ps=norm(photo[tr]);P/=np.sqrt(P.shape[1]);im=isc=None
    if X is not None:
        Z,im,isc=norm(X[tr]);Z/=np.sqrt(Z.shape[1]);Z=np.c_[Z,P]/np.sqrt(2)
    else:Z=P
    K=kernel(Z,Z);A=np.linalg.solve(K+alpha*np.eye(len(tr)),np.eye(len(tr)));v=A.sum(1);C=A-v[:,None]*v[None,:]/v.sum()
    return dict(image_mean=im,image_scale=isc,photo_mean=pm,photo_scale=ps,Z=Z,dual=C@y[tr],bias=float(v@y[tr]/v.sum()),alpha=alpha)
def predict(m,X,photo,q):
    P=np.clip((photo[q]-m['photo_mean'])/m['photo_scale'],-8,8)/np.sqrt(photo.shape[1])
    if X is not None:
        Z=np.clip((X[q]-m['image_mean'])/m['image_scale'],-8,8)/np.sqrt(X.shape[1]);Z=np.c_[Z,P]/np.sqrt(2)
    else:Z=P
    return m['bias']+kernel(m['Z'],Z)@m['dual']

def main():
    d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];ids=d['ids'][dev];y=d['y'][dev];fold=d['fold'];base=d['base083'];eligible=~d['failed']
    photo=np.load(_repo_path(HERE, 'microscopy-contrast/photometry.npz'));assert np.array_equal(photo['ids'],d['ids']);photo=photo['X'][dev]
    arms={'photometry':None}
    for arm in ['micronet','imagenet']:
        z=np.load(_repo_path(HERE, f'microscopy-contrast/{arm}.npz'));assert np.array_equal(z['ids'],d['ids'])
        for kind in ['raw','contrast']:arms[arm+'_'+kind]=z[kind][dev].astype(float)
    out=_repo_path(HERE, 'microscopy-contrast/readout');out.mkdir(exist_ok=True);alpha_grid=[.001,.01,.1,1.,10.];preds={};records=[];start=time.monotonic()
    spec=dict(kernel='Equal linear/RBFsum;equal geometry/photometryblockweight;per-training-blockRMSfloorstandardization;RBFexp(-distance²/2)',alpha_grid=alpha_grid,selection='Inner3fold refitsallnormalizationandhead;outeroriginal5folds.',primary_combination='Complete083+.16*(newimageprediction-matchedG021),existingfailedmaskrowsunchanged.',secondary='fixed25%blend',scope='Newimagepredictorend-to-endhasnoevaluation-imagelearning;full083comparisonconditionalonitshistoricalauxiliaryrepresentations.')
    (out/'recipe.json').write_text(json.dumps(spec,indent=2))
    p21=np.load(_repo_path(ROOT, 'experiments/rmse-research-20261015/inputs.npz'))['C73_cnn'][0]
    for name,X in arms.items():
        p=np.empty(400);details=[]
        for k in range(5):
            tr=np.flatnonzero(fold!=k);q=np.flatnonzero(fold==k);loss=np.zeros(len(alpha_grid))
            for a,b in KFold(3,shuffle=True,random_state=2026101921+k).split(tr):
                for j,alpha in enumerate(alpha_grid):
                    m=fit(X,photo,y,tr[a],alpha);loss[j]+=np.sum((predict(m,X,photo,tr[b])-y[tr[b]])**2)
            alpha=alpha_grid[int(np.argmin(loss))];m=fit(X,photo,y,tr,alpha);p[q]=predict(m,X,photo,q)
            arrays={key:value for key,value in m.items() if value is not None};np.savez(_repo_io(out/f'{name}-fold{k}.npz'),train=tr,query=q,prediction=p[q],**arrays);details.append(dict(fold=k,alpha=alpha,inner_rmse=np.sqrt(loss/len(tr)).tolist()))
        r=base+.16*eligible*(p-p21);b=base+.25*eligible*(p-base);rm=lambda a:float(np.sqrt(np.mean((a-y)**2)))
        rec=dict(arm=name,standalone_rmse=rm(p),G021_rmse=rm(p21),base083_rmse=rm(base),replacement_rmse=rm(r),quarter_blend_rmse=rm(b),replacement_fold_rmse=[float(np.sqrt(np.mean((r[fold==k]-y[fold==k])**2))) for k in range(5)],selection=details)
        records.append(rec);preds[name]=p;preds[name+'_replacement']=r;preds[name+'_blend']=b;print(json.dumps(rec),flush=True)
    np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,base083=base,p021=p21,**preds);(out/'results.json').write_text(json.dumps(dict(results=records,seconds=time.monotonic()-start),indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
