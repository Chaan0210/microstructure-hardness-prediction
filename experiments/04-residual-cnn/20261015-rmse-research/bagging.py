"""Frozen 16 x 90% GMM subbagging vs exact full-data and penalty-only controls."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
import research as r


def main():
    start=time.monotonic(); d=np.load(_repo_io(r.HERE/'inputs.npz')); dev=d['dev']; y=d['y'][dev]; fold=d['fold']
    path=r.EXP/'push-900/gmm_partner.py'
    # Reuse the existing per-image measurement and fitting functions; omit its CLI.
    namespace={'__file__':str(path),'__name__':'gmm_reuse'}
    exec(compile(path.read_text().split('mode = sys.argv[1]')[0],str(path),'exec'),namespace)
    bankpath=r.EXP/'probabilistic-grains-900/object-bank-train.npz'
    bank=np.load(_repo_io(bankpath),allow_pickle=True)
    xg=namespace['gmm_feats'](bank,list(d['ids'][dev]))
    np.savez_compressed(_repo_io(r.HERE/'gmm-features-dev.npz'),ids=d['ids'][dev],X=xg)
    prediction={key:np.zeros((3,len(dev))) for key in ('full','penalty_control','bagged')}; details=[]
    empty=np.zeros((0,2),int)
    for p in range(3):
        for k in range(5):
            train=np.where(fold[p]!=k)[0]; valid=np.where(fold[p]==k)[0]
            for noisy in (False,True):
                keep=np.ones(len(d['keys']),bool) if noisy else d['keep']
                x=np.c_[d['X'][dev][:,keep],xg]
                seed=19001+101*p+k if noisy else 19001
                full,alpha=namespace['fit_predict'](x[train],y[train],x[valid],seed)
                pcontrol=namespace['ridge'](alpha/.9).fit(x[train],y[train]).predict(x[valid])
                rng=np.random.default_rng(r.SEED+101*p+k); bags=[]
                for _ in range(16):
                    subset=rng.choice(train,int(.9*len(train)),replace=False)
                    assert not set(subset)&set(valid)
                    bags.append(namespace['ridge'](alpha).fit(x[subset],y[subset]).predict(x[valid]))
                bag=np.mean(bags,axis=0); mask=(d['noise'][dev[valid]]>=12)==noisy
                for key,values in [('full',full),('penalty_control',pcontrol),('bagged',bag)]:
                    prediction[key][p,valid[mask]]=values[mask]
                details.append(dict(partition=p,fold=k,noisy=noisy,alpha=float(alpha),seed=seed,bag_seed=r.SEED+101*p+k,n_train=len(train),n_per_bag=int(.9*len(train))))
            print('GMM bagging',p,k,'seconds',round(time.monotonic()-start,1),flush=True)
    diff=float(np.max(np.abs(prediction['full']-d['C73_gmm'])[~d['fail']]))
    assert diff<1e-8,diff
    blends={key:np.where(d['fail'],d['c73'],d['c73']+.15*(values-d['C73_gmm'])) for key,values in prediction.items()}
    result={key:r.compare(d['c73'],values,y,fold) for key,values in blends.items()}
    result['bagged_vs_penalty_control']=r.compare(blends['penalty_control'],blends['bagged'],y,fold)
    result['details']=details; result['full_reproduction_max_difference']=diff
    result['standalone_rmse']={key:r.rmse(values,y) for key,values in prediction.items()}
    result['seconds']=time.monotonic()-start
    result['input_hashes']={str(p.relative_to(r.ROOT)):r.digest(p) for p in [path,bankpath,r.HERE/'inputs.npz',r.HERE/'PLAN.md']}
    np.savez_compressed(_repo_io(r.HERE/'bagging.npz'),**prediction,**{key+'_blend':value for key,value in blends.items()})
    r.save_json(r.HERE/'bagging.json',result)
    print(json.dumps({k:result[k] for k in ['bagged','penalty_control','standalone_rmse','full_reproduction_max_difference','seconds']},indent=2))


if __name__=='__main__':
    if (r.HERE/'bagging.npz').exists(): raise FileExistsError('bagging.npz')
    with threadpool_limits(limits=2): main()
