"""H3 and exact source-scope sensitivity; no shared file mutations."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import json
import time
from collections import Counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import research as r


class NuisanceRidge(r.DomainRidge):
    def __init__(self,alpha,scale,keep):
        super().__init__(alpha,scale)
        self.keep=keep

    def design(self,z,domain):
        common=z[:,self.keep]
        if not self.scale: return common
        return np.c_[common, self.scale*z[:,~self.keep]*(domain==1)[:,None]]


def nuisance():
    start=time.monotonic(); d=np.load(_repo_io(r.HERE/'inputs.npz'))
    if (r.HERE/'nuisance.npz').exists(): raise FileExistsError('nuisance.npz')
    x,y,noise,dev,fold=[d[k] for k in ('X','y','noise','dev','fold')]
    preds={k:d['ridge73'].copy() for k in ('control','candidate')}
    grid=[(alpha,scale) for scale in (0.,.3,1.) for alpha in r.ALPHAS]
    choices=[]; pairs=np.zeros((0,2),int)
    for p in range(3):
        for k in range(5):
            train=dev[fold[p]!=k]; valid=dev[fold[p]==k]
            dom=(noise[train]>=12).astype(int)
            losses=np.zeros(len(grid)); n=0
            for a,b in GroupKFold(3,shuffle=True,random_state=r.SEED).split(x[train],y[train],train):
                b=b[dom[b]==0]
                for j,(alpha,scale) in enumerate(grid):
                    m=NuisanceRidge(alpha,scale,d['keep']).fit(x[train[a]],y[train[a]],dom[a],pairs)
                    losses[j]+=np.sum((m.predict(x[train[b]],dom[b])-y[train[b]])**2)
                n+=len(b)
            control=int(np.argmin(losses[:len(r.ALPHAS)])); candidate=int(np.argmin(losses))
            choices.append(dict(partition=p,fold=k,control=grid[control],candidate=grid[candidate],inner_mse=(losses/n).tolist()))
            ii=np.where((fold[p]==k)&(noise[dev]<12))[0]
            for name,j in [('control',control),('candidate',candidate)]:
                alpha,scale=grid[j]
                m=NuisanceRidge(alpha,scale,d['keep']).fit(x[train],y[train],dom,pairs)
                preds[name][p,ii]=m.predict(x[dev[ii]],np.zeros(len(ii),int))
    blends={k:np.where(d['fail'],d['c73'],d['c73']+.128*(v-d['ridge73'])) for k,v in preds.items()}
    result={k:r.compare(d['c73'],v,y[dev],fold) for k,v in blends.items()}
    result['candidate_vs_control']=r.compare(blends['control'],blends['candidate'],y[dev],fold)
    result['selections']=choices; result['seed']=r.SEED; result['seconds']=time.monotonic()-start
    result['code_sha256']=r.digest(__file__)
    np.savez_compressed(_repo_io(r.HERE/'nuisance.npz'),**preds,**{k+'_blend':v for k,v in blends.items()})
    r.save_json(r.HERE/'nuisance.json',result)
    print(json.dumps({k:result[k] for k in ('control','candidate','candidate_vs_control','seconds')},indent=2))


def scope():
    """Fixed four-bank logRidge: audit the effect of extra historical holdout sources."""
    d=np.load(_repo_io(r.HERE/'inputs.npz')); dev=d['dev']; fold=d['fold']; y=d['y']; ids=d['ids']
    if (r.HERE/'scope.npz').exists(): raise FileExistsError('scope.npz')
    pos={s:j for j,s in enumerate(ids)}; predictions={'historical':[],'strict':[]}; hashes={}
    for variant in ('u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z'):
        path=r.EXP/f'push-800/feat-{variant}-anghcor-train.npz'
        f=np.load(_repo_io(path),allow_pickle=True); hashes[str(path.relative_to(r.ROOT))]=r.digest(path)
        fi=list(f['ids'].astype(str)); x=f['X'].astype(float)[[fi.index(i) for i in ids]]
        keys=list(f['keys'].astype(str)); xs=[]; source=[]
        for tag in ('deg','deg2','deg3','deg4'):
            sp=r.EXP/f'push-1200/synthrows-{tag}-{variant}-anghcor.npz'
            s=np.load(_repo_io(sp),allow_pickle=True); hashes[str(sp.relative_to(r.ROOT))]=r.digest(sp)
            assert list(s['keys'].astype(str))==keys
            m=s['X'][:,keys.index('noise')].astype(float)*255>=12
            xs.extend(s['X'].astype(float)[m]); source.extend(pos[str(i)] for i in s['ids'][m])
        xs,source=np.array(xs),np.array(source)
        for mode in predictions:
            o=np.zeros((3,len(dev)))
            for p in range(3):
                for k in range(5):
                    tr,va=dev[fold[p]!=k],dev[fold[p]==k]
                    use=np.isin(source,tr) if mode=='strict' else ~np.isin(source,va)
                    assert not set(source[use])&set(va)
                    m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=2.))
                    m.fit(np.vstack([x[tr],xs[use]]),np.log(np.r_[y[tr],y[source[use]]]))
                    o[p,fold[p]==k]=m.predict(x[va])
            predictions[mode].append(o)
        print('completed',variant,flush=True)
    predictions={k:np.exp(np.mean(v,axis=0)) for k,v in predictions.items()}
    gated=(d['noise'][dev]>=12)&~d['fail'][0]
    ref=d['C73_logr']; delta=predictions['strict']-predictions['historical']
    old_r=np.load(_repo_io(r.HERE/'consistency.npz'))['production']
    strict_r=np.load(_repo_io(r.HERE/'consistency-strict.npz'))['production']
    old=d['c73']; new=old.copy(); new[:,gated]+=.128*(delta+strict_r-old_r)[:,gated]
    result=r.compare(old,new,y[dev],fold)
    result['historical_logridge_maxdiff_gated']=float(np.max(np.abs(predictions['historical']-ref)[:,gated]))
    result['protocol_note']='Strict excludes the 46 non-dev sources. This is a training-size sensitivity, not validation-fold leakage or a new submission candidate.'
    result['inputs_sha256']=hashes
    r.save_json(r.HERE/'scope.json',result)
    np.savez_compressed(_repo_io(r.HERE/'scope.npz'),**predictions,strict_blend=new)
    print(json.dumps({k:result[k] for k in ('candidate','gain','partition_gains','conditional_id_bootstrap_gain_ci95','historical_logridge_maxdiff_gated')},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['nuisance','scope']); args=ap.parse_args()
    with threadpool_limits(limits=2):
        nuisance() if args.action=='nuisance' else scope()
