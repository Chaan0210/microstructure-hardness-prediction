"""Frozen image recipe, matched local nonlinear readouts on its train-only features."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME']=str(_repo_path(HERE, 'tmp/huggingface'));os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import argparse,json,hashlib,time,gc
import numpy as np
import torch
from threadpoolctl import threadpool_limits

def augment(f):
    # Known heterogeneity × elongation relationships from the incumbent, fixed before new scores.
    return np.c_[f,f[:,[13,15,17,14,16,18]]*f[:,7:8]]

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));folder=_repo_path(HERE, 'runs', a.run)
    z=np.load(_repo_io(folder/f'{a.stage}-features.npz'));tr=z['train'];q=z['query'];assert not set(tr)&set(q)
    f=a.fold;ix=np.flatnonzero(d['fold']==f);assert np.array_equal(tr,d['dev'][d['fold']!=f]) and np.array_equal(q,d['dev'][ix])
    out=_repo_path(HERE, f'phase-readout-{a.run}-{a.stage}');out.mkdir(exist_ok=True)
    sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
    recipe=dict(run=a.run,stage=a.stage,fold=f,train_ids=d['ids'][tr].tolist(),validation_ids=d['ids'][q].tolist(),
                feature_file_sha256=hashlib.sha256((folder/f'{a.stage}-features.npz').read_bytes()).hexdigest(),
                seeds=[0,100,200],n_estimators=8,model='existing local TabPFNv3.5 checkpoint',device='CPU',
                features=[f"{z['train_features'].shape[1]}physical +{z['train_globals'].shape[1]}global",f"{z['train_features'].shape[1]}physical +6fixedinteractions +{z['train_globals'].shape[1]}global"],
                comparison='Complete083sameheldouts;fixed25%blend preserves existing failed-mask fallback. Existing baseline remains conditional on historical auxiliary representations.')
    (out/'recipe.json').write_text(json.dumps(recipe,indent=2));result={};allp={}
    variants=[('base_features',lambda x:x)] if 'includes_interactions' in z and bool(z['includes_interactions']) else [('base_features',lambda x:x),('interaction_features',augment)]
    for name,fn in variants:
        X=np.c_[fn(z['train_features']),z['train_globals']];Q=np.c_[fn(z['query_features']),z['query_globals']];ps=[]
        for seed in [0,100,200]:
            cache=out/f'{name}-seed{seed}.npz'
            if cache.exists():pr=np.load(_repo_io(cache))['prediction']
            else:
                t=time.monotonic();m=harness.tabpfn_new('v3.5',seed,'cpu',n_est=8);m.fit(X,d['y'][tr]);pr=m.predict(Q)
                np.savez(_repo_io(cache),train=tr,query=q,prediction=pr);print(name,seed,'seconds',round(time.monotonic()-t,1),flush=True);del m;gc.collect()
            ps.append(pr)
        p=np.mean(ps,0);base=d['base083'][ix];eligible=~d['failed'][ix];blend=base+.25*eligible*(p-base);y=d['y'][q]
        rm=lambda v:float(np.sqrt(np.mean((v-y)**2)))
        result[name]=dict(feature_count=int(X.shape[1]),rmse=rm(p),with_failure_fallback_rmse=rm(np.where(eligible,p,base)),base083_rmse=rm(base),quarter_blend_rmse=rm(blend),per_seed_rmse=[rm(v) for v in ps]);allp[name]=p;allp[name+'_blend']=blend
        print(name,json.dumps(result[name]),flush=True)
    np.savez(_repo_io(out/'predictions.npz'),train=tr,query=q,ids=d['ids'][q],y=d['y'][q],base083=d['base083'][ix],**allp)
    (out/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='phase-joint-b4-f0-s81');p.add_argument('--stage',choices=['frozen','joint'],default='frozen');p.add_argument('--fold',type=int,default=0)
    with threadpool_limits(limits=2):main(p.parse_args())
