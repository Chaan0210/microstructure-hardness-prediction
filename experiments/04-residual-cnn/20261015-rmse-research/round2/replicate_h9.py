"""Two additional seeds in all three H9 arms; no parameter/feature selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import gc
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from full500 import HERE,SEEDS,W,harness
from models import inputs
import research as r


def main():
    start=time.monotonic();d=inputs();ids=d['ids'];lookup={i:j for j,i in enumerate(ids)}
    c=np.load(_repo_io(HERE/'cumulants-train.npz'));extra=c['X'][:,[1,4]]
    random=np.random.default_rng(2026101509).normal(size=extra.shape)
    failed=np.nan_to_num(d['X'][:,list(d['keys']).index('n_obj')],nan=0)<20
    root=HERE/'h9-replicates';root.mkdir(exist_ok=True)
    for rep,seed in enumerate(SEEDS):
        for k in range(5):
            old=np.load(_repo_io(HERE/f'full500-{seed}/fold{k}.npz'));tr=np.array([lookup[i]for i in old['train_ids']]);va=old['validation_index']
            assert not set(tr)&set(va)
            for add in (100,200):
                path=root/f'part{rep}-fold{k}-seed{add}.npz'
                if path.exists():continue
                out={key:np.zeros(len(va))for key in ('base_tab','physical_tab','random_tab')}
                for noisy in (False,True):
                    cols=np.ones(len(d['keys']),bool)if noisy else d['keep'];mask=(d['noise'][va]>=12)==noisy;vi=va[mask]
                    for key,e in [('base_tab',None),('physical_tab',extra),('random_tab',random)]:
                        xt=d['X'][tr][:,cols];xv=d['X'][vi][:,cols]
                        if e is not None:xt=np.c_[xt,e[tr]];xv=np.c_[xv,e[vi]]
                        model=harness.tabpfn_new('v3.5',seed+k+add,'cpu',n_est=4)
                        model.fit(xt,d['y'][tr]);out[key][mask]=model.predict(xv)
                        del model;gc.collect()
                np.savez_compressed(_repo_io(path),train_ids=d['ids'][tr],validation_ids=d['ids'][va],validation_index=va,seed=seed+k+add,**out)
                print('H9 replicate',rep,k,add,'elapsed',round(time.monotonic()-start,1),flush=True)
    out={key:np.zeros((3,500))for key in ('base','physical','random')};fold=np.zeros((3,500),int)
    for rep,seed in enumerate(SEEDS):
        for k in range(5):
            old=np.load(_repo_io(HERE/f'full500-{seed}/fold{k}.npz'));h9=np.load(_repo_io(HERE/f'h9/part{rep}-fold{k}.npz'));va=old['validation_index']
            members={key:[value]for key,value in [('base',old['tab']),('physical',h9['physical']),('random',h9['random'])]}
            for add in (100,200):
                z=np.load(_repo_io(root/f'part{rep}-fold{k}-seed{add}.npz'))
                assert np.array_equal(z['validation_index'],va)
                for key in members:members[key].append(z[key+'_tab'])
            for key,values in members.items():
                p=old['base']+W[0]*(np.mean(values,axis=0)-old['tab']);p[failed[va]]=old['base'][failed[va]]
                out[key][rep,va]=p
            fold[rep,va]=k
    result={key:r.compare(out['base'],v,d['y'],fold)for key,v in out.items()if key!='base'}
    result['physical_vs_random']=r.compare(out['random'],out['physical'],d['y'],fold)
    one=np.load(_repo_io(HERE/'h9-summary.npz'))
    result['three_vs_one_seed_baseline']=r.compare(one['base'],out['base'],d['y'],fold)
    result['seed_offsets']=[0,100,200];result['seconds']=time.monotonic()-start
    result['code_sha256']=r.digest(__file__);result['plan_sha256']=r.digest(HERE/'PLAN-H9-REPLICATION.md')
    np.savez_compressed(_repo_io(HERE/'h9-three-seeds.npz'),ids=ids,fold=fold,**out)
    r.save_json(HERE/'h9-three-seeds.json',result)
    for key in ('physical','random','physical_vs_random'):
        q=result[key];print(key,'rmse',q['candidate']['pooled'],'gain',q['gain'],'parts',q['partition_gains'],'CI',q['conditional_id_bootstrap_gain_ci95'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
