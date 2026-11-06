"""Isolate native TabPFN readout memory; identical fixed predictors and seeds."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, sys
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[k] = '2'
for k in ['TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME']:
    os.environ[k] = str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME'] = str(_repo_path(HERE, 'tmp/huggingface'))
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
import argparse, gc, resource
import numpy as np
import torch
from threadpoolctl import threadpool_limits


def main(a):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    source = Path(a.input).resolve(); target = Path(a.output).resolve()
    assert source.is_relative_to(HERE) and target.is_relative_to(HERE)
    z = np.load(_repo_io(source)); X, y, Q = z['X'], z['y'], z['Q']; fold = int(z['fold'])
    sys.path.insert(0, str(_repo_path(ROOT, 'experiments/push-300'))); import harness
    ps = []
    for seed in [0, 100, 200]:
        m = harness.tabpfn_new('v3.5', seed+fold, 'cpu', n_est=8)
        m.fit(X, y); ps.append(m.predict(Q)); del m; gc.collect()
    np.savez(_repo_io(target), prediction=np.mean(ps, 0), peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    with threadpool_limits(2):
        main(p.parse_args())
