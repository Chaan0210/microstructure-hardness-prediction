"""H9 fixed phase/contrast pair, honest all-500 paired TabPFN experiment."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import gc
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from full500 import HERE,SEEDS,W,harness
from models import inputs
import research as r


def run():
    start=time.monotonic();d=inputs();ids=d['ids'];lookup={i:j for j,i in enumerate(ids)}
    c=np.load(_repo_io(HERE/'cumulants-train.npz'));assert np.array_equal(c['ids'],ids)
    extra=c['X'][:,[1,4]];random=np.random.default_rng(2026101509).normal(size=extra.shape)
    failed=np.nan_to_num(d['X'][:,list(d['keys']).index('n_obj')],nan=0)<20
    root=HERE/'h9';root.mkdir(exist_ok=True)
    for rep,seed in enumerate(SEEDS):
        for k in range(5):
            output=root/f'part{rep}-fold{k}.npz'
            if output.exists():continue
            old=np.load(_repo_io(HERE/f'full500-{seed}/fold{k}.npz'))
            tr=np.array([lookup[i]for i in old['train_ids']]);va=old['validation_index']
            assert not set(tr)&set(va)
            pred={name:np.zeros(len(va))for name in ('physical','random')}
            for noisy in (False,True):
                cols=np.ones(len(d['keys']),bool)if noisy else d['keep']
                mask=(d['noise'][va]>=12)==noisy;vi=va[mask]
                for name,e in [('physical',extra),('random',random)]:
                    model=harness.tabpfn_new('v3.5',seed+k,'cpu',n_est=4)
                    model.fit(np.c_[d['X'][tr][:,cols],e[tr]],d['y'][tr])
                    pred[name][mask]=model.predict(np.c_[d['X'][vi][:,cols],e[vi]])
                    del model;gc.collect()
            blends={name:old['base']+W[0]*(p-old['tab'])for name,p in pred.items()}
            for p in blends.values():p[failed[va]]=old['base'][failed[va]]
            np.savez_compressed(_repo_io(output),train_ids=d['ids'][tr],validation_ids=d['ids'][va],validation_index=va,
                base=old['base'],**pred,**{name+'_blend':p for name,p in blends.items()})
            print('H9',rep,k,'base',round(r.rmse(old['base'],d['y'][va]),5),'physical',round(r.rmse(blends['physical'],d['y'][va]),5),
                'random',round(r.rmse(blends['random'],d['y'][va]),5),'elapsed',round(time.monotonic()-start,1),flush=True)
    out={key:np.zeros((3,500))for key in ('base','physical_blend','random_blend')};fold=np.zeros((3,500),int)
    for rep in range(3):
        for k in range(5):
            z=np.load(_repo_io(root/f'part{rep}-fold{k}.npz'));ix=z['validation_index'];fold[rep,ix]=k
            for key in out:out[key][rep,ix]=z[key]
    results={key:r.compare(out['base'],out[key],d['y'],fold)for key in ('physical_blend','random_blend')}
    results['physical_vs_random']=r.compare(out['random_blend'],out['physical_blend'],d['y'],fold)
    results['seeds']=SEEDS;results['columns']=[str(c['keys'][j])for j in [1,4]];results['seconds']=time.monotonic()-start
    results['code_sha256']=r.digest(__file__);results['plan_sha256']=r.digest(HERE/'PLAN-H9.md')
    np.savez_compressed(_repo_io(HERE/'h9-summary.npz'),ids=ids,fold=fold,**out)
    r.save_json(HERE/'h9-summary.json',results)
    for key,q in results.items():
        if isinstance(q,dict):print(key,q['candidate']['pooled'],q['gain'],q['partition_gains'],q['conditional_id_bootstrap_gain_ci95'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):run()
