"""Reuse fixed scattering measurements with invariant normalized interactions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import hashlib,json,time,socket,gc
import numpy as np
import torch
from threadpoolctl import threadpool_limits

def reduced(X):
    first=X[:,1:49].reshape(-1,6,8);mean=first.mean(2)
    cols=[X[:,:1],np.log(np.maximum(mean,1e-8)),first.std(2)/np.maximum(mean,1e-8)]
    meta=[(j,t) for j in range(6) for t in range(8)]
    lookup={};column=49
    for j1,t1 in meta:
        for j2,t2 in meta:
            if j2>j1:lookup[(j1,j2,t1,t2)]=column;column+=1
    assert column==1009
    for j1 in range(6):
        for j2 in range(j1+1,6):
            v=np.stack([np.log(np.maximum(X[:,lookup[(j1,j2,t1,t2)]],1e-9)/np.maximum(first[:,j1,t1],1e-8)) for t1 in range(8) for t2 in range(8)],1)
            delta=np.array([t2-t1 for t1 in range(8) for t2 in range(8)])
            cols.append(np.stack([(v*np.cos(2*np.pi*h*delta/8)).mean(1) for h in [0,1,2]],1))
    return np.concatenate(cols,1)

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];ids=d['ids'][dev];y=d['y'][dev];fold=d['fold']
    path=_repo_path(ROOT, 'artifacts/wavelet-scattering-096/development_features.npz');record=json.loads((path.parent/'extraction.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['features_sha256']
    z=np.load(_repo_io(path));lookup={str(i):k for k,i in enumerate(z['ids'])};W=z['X'][[lookup[str(i)] for i in ids]];S=reduced(W)
    assert S.shape==(400,58) and np.isfinite(S).all()
    np.savez(_repo_path(HERE, 'scattering58.npz'),ids=ids,X=S)
    sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
    P=d['physical'][dev].copy();C=np.load(_repo_path(ROOT, 'experiments/push-1300/cumul-train.npz'))
    cm={str(i):k for k,i in enumerate(C['ids'])};cum=C['X'][[cm[str(i)] for i in ids]]
    # Keep the previously used sigma-1 pair, matching the physical100 readout.
    P=np.c_[P,cum[:,[1,4]]]
    pred={k:np.full(400,np.nan) for k in ['physical100','physical_scattering','scattering_only']};records=[]
    out=_repo_path(HERE, 'scattering-readout-v2');out.mkdir(exist_ok=True)
    (out/'recipe.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),feature_cache_sha256=record['features_sha256'],
        columns='58 deterministic rotation/reflection pooled log-scattering coefficients: S0,6 log S1 means,6 orientation CVs,15 scale pairs x3 relative-angle log(S2/S1) Fourier averages',
        seed=2026101831,n_estimators=4,device='cpu',baseline='physical100 is a paired readout control, NOT complete incumbent',
        screening='Original five folds, conditional on historical physical features. Fixed addition 083+0.1*(augmented TabPFN-physical100 TabPFN); no weight fit.'),indent=2))
    for f in range(5):
        tr=np.flatnonzero(fold!=f);va=np.flatnonzero(fold==f)
        for kind,X in [('physical100',P),('physical_scattering',np.c_[P,S]),('scattering_only',S)]:
            target=out/f'{kind}-fold{f}.npz'
            if target.exists():
                r=np.load(_repo_io(target));assert np.array_equal(r['train'],tr) and np.array_equal(r['validation'],va);p=r['prediction']
            else:
                t=time.monotonic();m=harness.tabpfn_new('v3.5',2026101831+f,'cpu',n_est=4);m.fit(X[tr],y[tr]);p=m.predict(X[va]);
                np.savez(_repo_io(target),train=tr,validation=va,train_ids=ids[tr],validation_ids=ids[va],prediction=p)
                print(kind,f,'seconds',round(time.monotonic()-t,1),'rmse',float(np.sqrt(np.mean((p-y[va])**2))),flush=True);del m;gc.collect()
            pred[kind][va]=p
    pred['fixed_addition']=d['base083']+.1*(pred['physical_scattering']-pred['physical100'])
    pred['fixed_scattering_blend']=.75*d['base083']+.25*pred['scattering_only']
    pred['base083']=d['base083'];rm=lambda a:float(np.sqrt(np.mean((a-y)**2)))
    result={k:dict(rmse=rm(v),fold_rmse=[float(np.sqrt(np.mean((v[fold==f]-y[fold==f])**2))) for f in range(5)]) for k,v in pred.items()}
    np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**pred);(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
