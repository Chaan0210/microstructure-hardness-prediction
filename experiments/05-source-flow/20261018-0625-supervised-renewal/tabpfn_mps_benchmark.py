"""Train-row-only numerical/runtime check; no validation score or test inputs."""
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
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
import fcntl, gc, json, time, resource, traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert torch.backends.mps.is_available()
    torch.mps.set_per_process_memory_fraction(.45)
    out = _repo_path(HERE, 'tabpfn-mps-benchmark'); out.mkdir(exist_ok=True)
    c = np.load(_repo_path(HERE, 'candidate-flow46-v1/training-context.npz'))
    X = np.c_[c['physical'], c['flow']]
    # Technical prediction queries are existing training rows. They are never
    # interpreted as held-out observations or used for quality/model selection.
    rows = np.unique(np.r_[np.linspace(0, 499, 100, dtype=int), 18, 59])
    Q = X[rows]
    state = _repo_path(HERE, 'candidate-flow46-v1/models/flow-seed0.tabpfn_fit')
    sys.path.insert(0, str(_repo_path(ROOT, 'experiments/push-300'))); import harness
    harness.tabpfn_new('v3.5', 0, 'cpu', n_est=8)
    from tabpfn import TabPFNRegressor
    results = {}; predictions = {}
    for device in ['cpu', 'mps']:
        model = TabPFNRegressor.load_from_fit_state(state, device=device)
        model.set_params(inference_precision=torch.float32)
        start = time.monotonic(); p = model.predict(Q)
        if device == 'mps':
            torch.mps.synchronize()
        results[device+'_seconds'] = time.monotonic()-start
        predictions[device] = p
        if device == 'mps':
            large = np.tile(Q, (int(np.ceil(1000/len(Q))), 1))[:1000]
            start = time.monotonic(); v = model.predict(large); torch.mps.synchronize()
            assert np.isfinite(v).all()
            results['mps_1000_train_queries_seconds'] = time.monotonic()-start
            results['mps_driver_bytes'] = torch.mps.driver_allocated_memory()
        del model; gc.collect()
    error = np.abs(predictions['cpu']-predictions['mps'])
    results.update(query_rows=len(Q), all_queries_are_training_rows=True, optimizer_updates=0,
                   test_inputs_read=False, quality_metric_computed=False,
                   numerical_max_difference=float(error.max()), numerical_rms_difference=float(np.mean(error**2)**.5),
                   peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   settings='Existing identical fitted state; CPU/MPS both explicitfloat32, native n8. Large benchmark repeats existing train rows only.')
    np.savez(_repo_io(out/'numerical-check.npz'), ids=c['ids'][rows], cpu=predictions['cpu'], mps=predictions['mps'])
    (out/'results.json').write_text(json.dumps(results, indent=2)); print('BENCHMARK', json.dumps(results), flush=True)


if __name__ == '__main__':
    with (_repo_path(HERE, 'tmp/heavy-learning.lock')).open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            with threadpool_limits(2):
                main()
        except Exception as error:
            out = _repo_path(HERE, 'tabpfn-mps-benchmark'); out.mkdir(exist_ok=True)
            (out/'failure.json').write_text(json.dumps(dict(error=repr(error), traceback=traceback.format_exc(),
                                                          test_inputs_read=False, optimizer_updates=0), indent=2))
            raise
