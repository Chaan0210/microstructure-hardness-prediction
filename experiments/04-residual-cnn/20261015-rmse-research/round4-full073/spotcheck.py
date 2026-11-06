"""Independent repeat of one exact TabPFN member and its input immutability."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import full_branch as b


def main():
    d,y=b.load();anchor=np.load(_repo_io(b.HERE/'anchors/fold0.npz'));tr,query=anchor['A'],anchor['B']
    x,cols=d['banks']['038','u1dm1szH2Z'];qi=query[d['noise'][query]<12]
    xx=x[tr][:,cols].copy();yy=y[tr].copy();xx_before=xx.copy();yy_before=yy.copy()
    model=b.harness.tabpfn_new('v3.5',0,'cpu');model.fit(xx,yy);pred=model.predict(x[qi][:,cols])
    assert np.array_equal(xx,xx_before,equal_nan=True)
    assert np.array_equal(yy,yy_before)
    cached=np.load(_repo_io(b.HERE/'teachers/fold0-outer0/tab-0-038-u1dm1szH2Z-seed0.npz'))
    assert np.array_equal(cached['train'],tr) and np.array_equal(cached['query'],qi)
    delta=float(np.max(np.abs(pred-cached['prediction'])))
    assert delta<1e-4,delta
    report=dict(max_abs_repeat_error=delta,training_features_unchanged=True,training_labels_unchanged=True,
                n_train=len(tr),n_query=len(qi),seed=0,requested_n_estimators=str(model.n_estimators))
    b.r.save_json(b.HERE/'tabpfn-independent-repeat.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
