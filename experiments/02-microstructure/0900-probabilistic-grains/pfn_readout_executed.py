"""Fixed strong readout control for fold-isolated probabilistic measurements.

Run only after measurement stage completes. Original features vs added posterior
phase mean vs mean+SD, all with same local pretrained TabPFN3.5 and seed/budget.
No hyperparameter or blend choice is made from outer results.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
for k,v in {'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2',
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TABPFN_DISABLE_TELEMETRY':'1',
            'XDG_CACHE_HOME':str(_repo_path(HERE, 'pfn-readout/cache')),'HF_HOME':str(_repo_path(HERE, 'pfn-readout/cache/hf')),
            'TORCH_HOME':str(_repo_path(HERE, 'pfn-readout/cache/torch'))}.items():os.environ.setdefault(k,v)
import argparse
import json
import socket
import time
import numpy as np
import torch
from common import ROOT,load_base,evaluate,destination,dump_json,sha

CONFIG={'model':'TabPFN3.5 local published checkpoint','n_estimators':8,'seeds':'fold index + 1000*partition',
        'arms':['base98','phase_mean','phase_mean_and_sd'],'added_columns':'posterior realized area phase fraction, then posterior SD',
        'blend':.2,'hyperparameter_search':False,'source':'fold-isolated measurement caches; fixed historical X98; conditional validation'}


def make(seed,device):
    def blocked(*a,**k):raise RuntimeError('Network disabled for local-only experiment')
    socket.create_connection=blocked;socket.getaddrinfo=blocked
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    path=_repo_path(ROOT, 'experiments/tabpfn-v3/weights/tabpfn-v3.5-20260909.safetensors')
    assert path.exists()
    return TabPFNRegressor.create_default_for_version(ModelVersion('v3.5'),model_path=str(path),device=device,
        random_state=seed,n_estimators=CONFIG['n_estimators'],show_progress_bar=False)


def augment(X,mean,var,arm):
    if arm=='base98':return X.copy()
    if arm=='phase_mean':return np.c_[X,mean[:,0]]
    return np.c_[X,mean[:,0],np.sqrt(np.maximum(var[:,0],0))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,default=_repo_path(HERE, 'probabilistic/stage2/features'))
    p.add_argument('--output',type=Path,default=_repo_path(HERE, 'pfn-readout'));p.add_argument('--device',default='mps');p.add_argument('--benchmark',action='store_true')
    p.add_argument('--wait-for-features',action='store_true');a=p.parse_args()
    out=destination(a.output/'placeholder').parent;b=load_base();torch.set_num_threads(2)
    dump_json(out/'CONFIG.json',{**CONFIG,'features':str(a.features),'device':a.device,'script_sha256':sha(__file__),'base_sha256':sha(_repo_path(HERE, 'base.npz'))})
    if a.benchmark:
        tr=np.flatnonzero(b['fold'][0]!=0)[:96]
        # Predict training-context rows purely to measure runtime; no heldout y.
        t=time.time();m=make(0,a.device);m.fit(b['X98'][tr],b['y'][tr]);pr=m.predict(b['X98'][tr[:8]])
        assert np.isfinite(pr).all();r={'seconds':time.time()-t,'device':a.device,'train_only_benchmark':True,'n_training':len(tr),'n_queries':8}
        dump_json(out/'benchmark.json',r);print(json.dumps(r),flush=True);return
    predictions={k:np.full_like(b['baseline'],np.nan) for k in CONFIG['arms']};records=[];t=time.time()
    for s in range(3):
        for f in range(5):
            cachepath=a.features/f'partition{s}-fold{f}-outer.npz'
            # A measurement pickle is written only after the NPZ has closed.
            marker=a.features/f'partition{s}-fold{f}-outer-measurement.pkl'
            if a.wait_for_features:
                wait_start=time.time()
                while not (cachepath.exists() and marker.exists()):
                    if time.time()-wait_start>3600:raise TimeoutError(f'Measurement cache unavailable: {cachepath}')
                    time.sleep(2)
            with np.load(_repo_io(cachepath),allow_pickle=False) as d:c={k:d[k] for k in d.files}
            tr,va=c['train_idx'],c['eval_idx'];assert np.array_equal(tr,np.flatnonzero(b['fold'][s]!=f));assert np.array_equal(va,np.flatnonzero(b['fold'][s]==f))
            record={'partition':s,'fold':f,'train_ids':b['ids'][tr].tolist(),'validation_ids':b['ids'][va].tolist(),'feature_sha256':sha(cachepath)}
            for arm in CONFIG['arms']:
                X=augment(b['X98'][tr],c['hierarchical_train_mean'],c['hierarchical_train_var'],arm)
                V=augment(b['X98'][va],c['hierarchical_eval_mean'],c['hierarchical_eval_var'],arm)
                model=make(1000*s+f,a.device);model.fit(X,b['y'][tr]);pr=model.predict(V)
                assert np.isfinite(pr).all();predictions[arm][s,va]=pr
                # Persist the complete train context and query transformations; weights are immutable local checkpoint.
                np.savez_compressed(_repo_io(destination(out/f'partition{s}-fold{f}-{arm}-context.npz')),train_ids=b['ids'][tr],query_ids=b['ids'][va],X=X,y=b['y'][tr],query=V,prediction=pr,seed=1000*s+f)
                del model
                if a.device=='mps':torch.mps.empty_cache()
            records.append(record);dump_json(out/'folds.json',records)
            np.savez_compressed(_repo_io(destination(out/'predictions-partial.npz')),ids=b['ids'],y=b['y'],**predictions)
            print(f'PFN fold{s}/{f} complete elapsed={time.time()-t:.1f}s',flush=True)
    results={}
    for arm,pr in predictions.items():
        q=np.where(b['failed'],b['p021'],pr)
        results[arm]={'standalone':evaluate(q,b),'blend20':evaluate(.8*b['baseline']+.2*q,b),
                     'same021':evaluate(.2*b['p021']+.8*(.8*b['grain']+.2*q),b)}
    dump_json(out/'results.json',{'arms':results,'seconds':time.time()-t,'config':CONFIG})
    np.savez_compressed(_repo_io(destination(out/'predictions.npz')),ids=b['ids'],y=b['y'],**predictions)
    print(json.dumps({k:{n:v[n]['rmse'] for n in v} for k,v in results.items()}),flush=True)


if __name__=='__main__':main()
