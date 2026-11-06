"""Matched readout for continuous multiscale flow geometry; train data only."""
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
import argparse, gc, hashlib, json, time
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def main(a):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    d = np.load(_repo_path(HERE, 'data.npz')); dev = d['dev']; fold = d['fold']; y = d['y'][dev]
    p = np.load(_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz'))
    pos = {str(v): i for i, v in enumerate(p['ids'])}
    P = p['X'][[pos[str(v)] for v in d['ids'][dev]]].astype(float)
    f = np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz'))
    z = np.load(_repo_path(HERE, 'flow-geometry/full-features.npz'))
    assert np.array_equal(f['ids'], d['ids']) and np.array_equal(z['ids'], d['ids'])
    X = np.c_[P, f['flow'][dev], z['geometry'][dev]]
    out = _repo_path(HERE, 'flow-geometry/readout'); out.mkdir(exist_ok=True)
    spec = dict(hypothesis='Continuous spatial grain coherence and phase-size covariance beyond marginal flow fields.',
                full_system='083 + eligible*(.384*(newTab-P104Tab) + .128*(newRidge-P104Ridge) + .128*(newLogRidge-P104LogRidge))',
                control='Frozen flow46 combined system, matched training rows and head seeds.',
                tab='Native local TabPFN3.5 n8, seeds0/100/200+fold', linear='Median imputer, StandardScaler, Ridge100 raw/log, exp inverse.',
                scope='Conditional globally trained H300/P104 representations; no claim of untouched or complete end-to-end validation.',
                feature_sha256=hashlib.sha256((_repo_path(HERE, 'flow-geometry/full-features.npz')).read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(spec, indent=2))
    sys.path.insert(0, str(_repo_path(ROOT, 'experiments/push-300'))); import harness
    oldtab = np.load(_repo_path(HERE, 'domain-semantics/readout/oof.npz'))['physical']
    lin = np.load(_repo_path(HERE, 'flow-linear-check/oof.npz'))
    records = []
    for k in a.folds:
        tr = np.flatnonzero(fold != k); q = np.flatnonzero(fold == k)
        ps = []
        for seed in [0, 100, 200]:
            path = out/f'f{k}-s{seed}.npz'
            if path.exists():
                c = np.load(_repo_io(path))
                assert np.array_equal(c['train'], tr) and np.array_equal(c['query'], q)
                v = c['prediction']
            else:
                start = time.monotonic()
                m = harness.tabpfn_new('v3.5', seed+k, 'cpu', n_est=8)
                m.fit(X[tr], y[tr]); v = m.predict(X[q]); del m; gc.collect()
                np.savez(_repo_io(path), train=tr, query=q, prediction=v)
                print('TAB', k, seed, time.monotonic()-start, flush=True)
            ps.append(v)
        tab = np.mean(ps, 0); ls = []
        for kind in ['raw', 'log']:
            m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=100))
            m.fit(X[tr], y[tr] if kind == 'raw' else np.log(y[tr]))
            v = m.predict(X[q]); ls.append(v if kind == 'raw' else np.exp(v))
        linear = np.mean(ls, 0)
        base = d['base083'][q]
        new = base+(~d['failed'][q])*(.384*(tab-oldtab[q])+.256*(linear-lin['control'][q]))
        control = lin['combined_system'][q]
        rm = lambda v: float(np.mean((v-y[q])**2)**.5)
        r = dict(fold=k, tab_rmse=rm(tab), linear_rmse=rm(linear), complete_system_rmse=rm(new),
                 frozen_flow46_system_rmse=rm(control), base083_rmse=rm(base), gain_over_flow46=rm(control)-rm(new))
        np.savez(_repo_io(out/f'predictions-f{k}.npz'), ids=d['ids'][dev[q]], train=tr, query=q, y=y[q], tab=tab, linear=linear, system=new, control=control, base083=base)
        (out/f'results-f{k}.json').write_text(json.dumps(r, indent=2))
        print('FOLD', json.dumps(r), flush=True); records.append(r)
    if set(a.folds) == set(range(5)):
        zz = [np.load(_repo_io(out/f'predictions-f{k}.npz')) for k in range(5)]
        yt = np.concatenate([z['y'] for z in zz]); pred = np.concatenate([z['system'] for z in zz]); control = np.concatenate([z['control'] for z in zz]); base = np.concatenate([z['base083'] for z in zz])
        r = dict(folds=records, rmse=float(np.mean((pred-yt)**2)**.5), control_rmse=float(np.mean((control-yt)**2)**.5), base083_rmse=float(np.mean((base-yt)**2)**.5), full_system_protocol_passed=False)
        (out/'results.json').write_text(json.dumps(r, indent=2)); print('DONE', json.dumps(r), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    with threadpool_limits(2):
        main(p.parse_args())
