"""Audit the installed library's frozen-model reuse; no fit-data reuse."""
import gc
import json
import os
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import full_branch as b


def main():
    d,y=b.load();records=[]
    cases=[(0,'outer0','038','u1dm1szH2Z',0,False),(0,'deep0','044','u1dmavgH2Z',1,True),
           (1,'outer1','039','u1dmavg4H2Z',2,False)]
    def run_case(case,cache,tag):
        fold,node,slot,variant,s,noisy=case
        file=b.HERE/f'teachers/fold{fold}-{node}/tab-{int(noisy)}-{slot}-{variant}-seed{s}.npz'
        z=np.load(file);tr=z['train'];query=z['query'];x,keep=d['banks'][slot,variant]
        cols=np.ones(x.shape[1],bool) if noisy else keep
        xx=x[tr][:,cols].copy();yy=y[tr].copy();xx0=xx.copy();yy0=yy.copy()
        os.environ['TABPFN_MODEL_CACHE_SIZE']=str(cache)
        start=time.monotonic();m=b.harness.tabpfn_new('v3.5',100*s+fold,'cpu')
        m.fit(xx,yy);p=m.predict(x[query][:,cols]);seconds=time.monotonic()-start
        assert np.array_equal(xx,xx0,equal_nan=True) and np.array_equal(yy,yy0)
        delta=float(np.max(np.abs(p-z['prediction'])));assert delta<1e-4,(tag,delta)
        record=dict(tag=tag,cache_size=cache,fold=fold,node=node,slot=slot,variant=variant,seed=100*s+fold,
                    n_train=len(tr),n_features=len(cols.nonzero()[0]),n_query=len(query),
                    max_abs_error_vs_original=delta,seconds=seconds,resolved_estimators=m.n_estimators_,
                    frozen_model_identity=id(m.models_[0]),training_inputs_unchanged=True)
        records.append(record);print(json.dumps(record),flush=True)
        del m;gc.collect()
    run_case(cases[0],0,'cold_reference')
    run_case(cases[0],1,'cache_initialization')
    run_case(cases[1],1,'different_shape_and_training_subset')
    run_case(cases[2],1,'different_fold_and_seed')
    # Deliberately populate a previous estimator with unrelated synthetic targets.
    # Its successor must reproduce the original current-fold prediction exactly.
    os.environ['TABPFN_MODEL_CACHE_SIZE']='1'
    xx=d['X'][d['dev'][:80]].copy();yy=10000.+17.*np.arange(80)
    poison=b.harness.tabpfn_new('v3.5',999,'cpu');poison.fit(xx,yy)
    del poison;gc.collect()
    run_case(cases[0],1,'after_poisoned_previous_fit')
    from tabpfn import model_loading
    model_loading.clear_built_model_cache();gc.collect()
    run_case(cases[0],0,'cold_after_library_warmup')
    result=dict(passed=True,records=records,previous_fit_context_poison_test=True,
                scope='Only immutable pretrained network construction is cached. Each estimator, preprocessing and training context remains fresh.',
                library_source=str(model_loading.__file__),library_sha256=b.r.digest(model_loading.__file__))
    b.r.save_json(b.HERE/'model-cache-audit.json',result)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
