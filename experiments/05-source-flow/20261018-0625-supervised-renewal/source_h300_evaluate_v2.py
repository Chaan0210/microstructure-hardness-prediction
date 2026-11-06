"""Complete conditional system comparison with a source-local new flow encoder."""
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
import argparse, gc, json
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def main(a):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    out = _repo_path(HERE, 'source-h300', a.tag); r = json.loads((out/'recipe.json').read_text())
    assert r['protocol'] == 'original400' and not set(r['source_indices']) & set(r['query'])
    d = np.load(_repo_path(HERE, 'data.npz')); dev = d['dev']; k = r['fold']; fold = d['fold']; y = d['y'][dev]
    tr = np.flatnonzero(fold != k); q = np.flatnonzero(fold == k)
    assert np.array_equal(dev[tr], r['train']) and np.array_equal(dev[q], r['query'])
    noise = np.load(_repo_path(ROOT, 'experiments/push-300/noise_train.npy'))
    expected = dev[tr][np.argsort(noise[dev[tr]], kind='stable')[:len(tr)//2]]
    assert np.array_equal(expected, r['source_indices'])
    z = np.load(_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz'))
    pos = {str(v): i for i, v in enumerate(z['ids'])}; P = z['X'][[pos[str(v)] for v in d['ids'][dev]]].astype(float)
    z = np.load(_repo_io(out/'flow-features.npz')); pos = {str(v): i for i, v in enumerate(z['ids'])}
    F = z['flow'][[pos[str(v)] for v in d['ids'][dev]]]; X = np.c_[P, F]
    sys.path.insert(0, str(_repo_path(ROOT, 'experiments/push-300'))); import harness
    ps, controls = [], []
    for seed in [0, 100, 200]:
        c = np.load(_repo_path(HERE, f'conductivity/readout/physical-f{k}-s{seed}.npz'))
        assert np.array_equal(c['train'], tr) and np.array_equal(c['query'], q); controls.append(c['prediction'])
        path = out/f'head-seed{seed}.npz'
        if path.exists():
            c = np.load(_repo_io(path)); assert np.array_equal(c['train'], tr) and np.array_equal(c['query'], q); v = c['prediction']
        else:
            m = harness.tabpfn_new('v3.5', seed+k, 'cpu', n_est=8); m.fit(X[tr], y[tr]); v = m.predict(X[q]); del m; gc.collect()
            np.savez(_repo_io(path), train=tr, query=q, prediction=v)
        ps.append(v)
    tab = np.mean(ps, 0); control = np.mean(controls, 0); linear = []
    for kind in ['raw', 'log']:
        m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=100))
        m.fit(X[tr], y[tr] if kind == 'raw' else np.log(y[tr])); v = m.predict(X[q]); linear.append(v if kind == 'raw' else np.exp(v))
    linear = np.mean(linear, 0); old = np.load(_repo_path(HERE, 'flow-linear-check/oof.npz'))
    base = d['base083'][q]; eligible = ~d['failed'][q]
    tab_system = base+.384*eligible*(tab-control)
    system = tab_system+.256*eligible*(linear-old['control'][q])
    rm = lambda v: float(np.mean((v-y[q])**2)**.5)
    result = dict(fold=k, source_rows=len(expected), query_source_overlap=0, ranking_used_only_train=True,
                  component_control_rmse=rm(control), component_flow_rmse=rm(tab), tab_system_rmse=rm(tab_system),
                  combined_system_rmse=rm(system), global_flow46_system_rmse=rm(old['combined_system'][q]), base083_rmse=rm(base),
                  complete_conditional_incumbent_compared=True, full_system_protocol_passed=False,
                  scope='New encoder and auxiliary eligibility are source-local. Inherited P104/083 global auxiliary representations remain conditional. Original400 development, not untouched data or complete end-to-end confirmation.')
    np.savez(_repo_io(out/'system-predictions.npz'), ids=d['ids'][dev[q]], query=dev[q], y=y[q], tab=tab, control=control, linear=linear, tab_system=tab_system, system=system, global_flow46=old['combined_system'][q], base083=base)
    (out/'system-results.json').write_text(json.dumps(result, indent=2)); print('SOURCE_SYSTEM', json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True)
    with threadpool_limits(2):
        main(p.parse_args())
