"""Bounded one-member throughput check before resuming the cached pilot."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import hashlib
import json
import signal
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import probe as p
import full_branch as b


def main():
    p.protect();p.sys.addaudithook(p.old.no_test)
    def timeout(*unused):raise TimeoutError('One-member60second resource probe limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
    d,y=b.load();g=json.loads((p.R4/'label-graph.json').read_text())[0]
    node=next(n for n in g['nodes'] if n['name']=='deep0')
    tr=np.array(node['train']);query=np.array(node['query'])
    path=p.HERE/'teachers076/fold0-deep0';contract=json.loads((path/'contract.json').read_text())
    assert contract['target_sha256']==hashlib.sha256(y[tr].tobytes()).hexdigest()
    c=np.load(_repo_path(p.ROOT, 'experiments/push-1300/cumul-train.npz'));cp={str(i):j for j,i in enumerate(c['ids'])}
    extra=c['X'][[cp[str(i)] for i in d['ids']]][:,[1,4]]
    for noisy in (False,True):
        qi=query[(d['noise'][query]>=12)==noisy]
        for (slot,variant),(raw,nbcols) in d['banks'].items():
            X=np.c_[raw,extra];cols=np.ones(X.shape[1],bool) if noisy else np.r_[nbcols,True,True]
            for seed in range(3):
                file=path/f'tab-{int(noisy)}-{slot}-{variant}-seed{seed}.npz'
                if file.exists():continue
                start=time.monotonic();model=b.harness.tabpfn_new('v3.5',100*seed,'cpu')
                model.fit(X[tr][:,cols],y[tr].copy());pred=model.predict(X[qi][:,cols])
                assert np.isfinite(pred).all();seconds=time.monotonic()-start
                np.savez_compressed(_repo_io(file),train=tr,query=qi,prediction=pred,seed=100*seed)
                signal.alarm(0)
                p.save(_repo_io('resource-probe.json'),dict(complete=True,seconds=seconds,member=str(file.relative_to(p.HERE)),
                    original_teacher_code_unchanged=True,source_contract_sha256=p.r.digest(path/'contract.json'),
                    resume_allowed_by_cost=(seconds<15),model_parameters_unchanged=True))
                print('ONE MEMBER COMPLETE',seconds,'seconds',flush=True);return
    signal.alarm(0);raise RuntimeError('No missing member to probe')


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
