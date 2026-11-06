"""Verify the installed library's built-model cache across seeds/data/feature gates."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
os.environ['TABPFN_MODEL_CACHE_SIZE']='1'
import gc
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import probe as p
import full_branch as b


def main():
    p.protect();p.sys.addaudithook(p.old.no_test);d,y=b.load()
    c=np.load(_repo_path(p.ROOT, 'experiments/push-1300/cumul-train.npz'));cp={str(i):j for j,i in enumerate(c['ids'])}
    extra=c['X'][[cp[str(i)] for i in d['ids']]][:,[1,4]]
    cases=[('outer0',0,'038',b.VARIANTS[0],0),('deep0',1,'044',b.VARIANTS[2],2),
           ('outer0',1,'039',b.VARIANTS[3],1),('deep0',0,'044',b.VARIANTS[1],0)]
    records=[];model_ids=[]
    for node,gate,slot,variant,seed in cases:
        path=p.HERE/f'teachers076/fold0-{node}/tab-{gate}-{slot}-{variant}-seed{seed}.npz'
        z=np.load(_repo_io(path));tr,query=z['train'],z['query'];raw,keep=d['banks'][slot,variant]
        X=np.c_[raw,extra];cols=np.ones(X.shape[1],bool) if gate else np.r_[keep,True,True]
        start=time.monotonic();model=b.harness.tabpfn_new('v3.5',100*seed,'cpu')
        model.fit(X[tr][:,cols],y[tr].copy());pred=model.predict(X[query][:,cols])
        elapsed=time.monotonic()-start;error=float(np.max(np.abs(pred-z['prediction'])))
        assert error==0.,(node,gate,slot,variant,seed,error)
        model_ids.append(id(model.models_[0]))
        records.append(dict(node=node,gate=gate,slot=slot,variant=variant,seed=seed,
                            exact_replay=True,max_error=error,seconds=elapsed,n_estimators=int(model.n_estimators_)))
        print('BUILT CACHE MATCH',node,gate,slot,seed,elapsed,flush=True)
        del model;gc.collect()
    assert len(set(model_ids))==1
    import tabpfn.model_loading as ml
    p.save(_repo_io('built-cache-verification.json'),dict(all_passed=True,cases=records,same_pretrained_model_object_reused=True,
        package_model_loading_sha256=p.r.digest(ml.__file__),environment_override={'TABPFN_MODEL_CACHE_SIZE':'1'},
        inference_only_no_foundation_weight_updates=True,teacher_algorithm_changed=False))
    p.protect()


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
