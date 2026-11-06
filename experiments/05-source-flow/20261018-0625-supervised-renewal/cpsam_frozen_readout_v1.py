"""Cheap controls for the unchanged pretrained descriptor, before fine-tuning."""
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
import gc,json,time
import numpy as np
import torch,joblib
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];fold=d['fold'];y=d['y'][dev];ids=d['ids'][dev]
    assert np.load(_repo_path(HERE, 'cpsam-prefix/done.npy')).all()
    F=np.load(_repo_path(HERE, 'cpsam-prefix/frozen-descriptors.npy')).mean(1)[dev]
    pixels=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');raw=[]
    for j in dev:
        x=(pixels[j].astype(float)/255-.5)*4;raw.append([float(np.mean(x**k)) for k in [1,2,3,4]])
    raw=np.array(raw);Z=np.c_[F,raw]
    C=np.load(_repo_path(ROOT, 'experiments/push-1300/cumul-train.npz'));index={str(i):j for j,i in enumerate(C['ids'])};P=np.c_[d['physical'][dev],C['X'][[index[str(i)] for i in ids]][:,[1,4]]]
    sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
    out=_repo_path(HERE, 'cpsam-frozen-readout');out.mkdir(exist_ok=True)
    predictions={k:np.full(len(dev),np.nan) for k in ['ridge','pca32_tab','physical100','physical_plus_cp']}
    for f in range(5):
        tr=np.flatnonzero(fold!=f);va=np.flatnonzero(fold==f)
        # Every alpha-selection fold fits its own scaler.
        m=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':[1,10,100,1000,10000]},
            cv=KFold(3,shuffle=True,random_state=2026101837+f),scoring='neg_mean_squared_error',n_jobs=1)
        m.fit(Z[tr],y[tr]);predictions['ridge'][va]=m.predict(Z[va]);joblib.dump(m,_repo_io(out/f'ridge-fold{f}.joblib'))
        pp=make_pipeline(StandardScaler(),PCA(32,svd_solver='full'));pp.fit(F[tr]);pc=pp.transform(F);joblib.dump(pp,_repo_io(out/f'pca-fold{f}.joblib'))
        for name,X in [('pca32_tab',np.c_[pc,raw]),('physical100',P),('physical_plus_cp',np.c_[P,pc,raw])]:
            t=time.monotonic();model=harness.tabpfn_new('v3.5',2026101831+f,'cpu',n_est=4)
            model.fit(X[tr],y[tr]);predictions[name][va]=model.predict(X[va]);del model;gc.collect()
            print(name,f,'seconds',round(time.monotonic()-t,1),'rmse',float(np.sqrt(np.mean((predictions[name][va]-y[va])**2))),flush=True)
        np.savez(_repo_io(out/f'fold{f}.npz'),train=tr,validation=va,train_ids=ids[tr],validation_ids=ids[va],**{k:v[va] for k,v in predictions.items()})
    predictions['base083']=d['base083']
    predictions['fixed_addition']=d['base083']+.1*(~d['failed'])*(predictions['physical_plus_cp']-predictions['physical100'])
    for name in ['ridge','pca32_tab']:
        predictions[name+'_quarter_blend']=.75*d['base083']+.25*predictions[name]
    scores={k:dict(rmse=float(np.sqrt(np.mean((p-y)**2))),fold_rmse=[float(np.sqrt(np.mean((p[fold==f]-y[fold==f])**2))) for f in range(5)]) for k,p in predictions.items()}
    np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**predictions)
    (out/'results.json').write_text(json.dumps(scores,indent=2));print(json.dumps(scores),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
