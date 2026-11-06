"""One actual PLS readout replay with held-out label poisoning."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import readout as r
from verify_readout import predict
import numpy as np,json,sys,time,torch
def main():
    start=time.monotonic();out=e.HERE/'alignment-followup';folder=out/'readouts/fold0-selector/spatial-seed0';c=np.load(_repo_io(folder/'context.npz'));old=np.load(_repo_io(folder/'tab0/prediction.npz'))['prediction'];d=r.physical();outside=np.setdiff1d(np.arange(500),c['train']);y=d['y'].copy();y[outside]+=10000
    assert np.array_equal(y[c['train']],c['y']);mps,small=predict(c['X'],y[c['train']],c['Q'],0,'mps');err=float(np.max(abs(mps-old)));subset=float(np.max(abs(small-mps[:8])));assert err<1e-5 and subset<1e-4
    cpu,_=predict(c['X'],y[c['train']],c['Q'][:8],0,'cpu');cpuerr=float(np.max(abs(cpu-mps[:8])));assert cpuerr<1e-3
    e.save(_repo_io('alignment-followup/numerical-replay.json'),dict(all_passed=True,context='fold0-selector/spatial-seed0/tab0',MPS_replay_max_HV_error=err,query_subset_error=subset,CPU_vs_MPS_max_HV_error=cpuerr,outside_labels_poisoned=len(outside),seconds=time.monotonic()-start,source_sha256=e.sha(__file__),scope='One selected context replay; all180 context/seed aggregates and60 PLS projections are separately audited.'))
    print('PLS READOUT REPLAY PASSED',err,subset,cpuerr,flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
