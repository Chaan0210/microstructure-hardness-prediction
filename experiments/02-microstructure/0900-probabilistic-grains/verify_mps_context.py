"""Replay one saved probabilistic-feature PFN context on the same MPS device."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(_repo_path(HERE, 'pfn-readout-v2/mps-replay/matplotlib'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import json
import numpy as np
import torch
from pfn_readout import make
from common import destination,dump_json,sha


def main():
    torch.set_num_threads(2)
    folder=destination(_repo_path(HERE, 'pfn-readout-v2/mps-replay/placeholder')).parent
    source=_repo_path(HERE, 'pfn-readout-v2/partition0-fold0-phase_mean_and_sd-context.npz')
    d=dict(np.load(_repo_io(source),allow_pickle=False))
    assert not set(d['train_ids']).intersection(d['query_ids'])
    m=make(int(d['seed']),'mps');m.fit(d['X'],d['y']);p=m.predict(d['query'])
    diff=p-d['prediction'];r={'source':str(source),'source_sha256':sha(source),'device':'mps',
        'n_train':len(d['train_ids']),'n_queries':len(d['query_ids']),'max_abs':float(np.max(np.abs(diff))),
        'rms':float(np.sqrt(np.mean(diff*diff))),'exact':bool(np.array_equal(p,d['prediction'])),
        'within_005_HV':bool(np.max(np.abs(diff))<.005),'no_query_labels_read':True}
    np.savez_compressed(_repo_io(folder/'predictions.npz'),replay=p,saved=d['prediction'])
    dump_json(folder/'verification.json',r);print(json.dumps(r),flush=True)
    assert r['within_005_HV'],'Same-device context replay failed tolerance'


if __name__=='__main__':main()
