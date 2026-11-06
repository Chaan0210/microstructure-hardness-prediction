"""Frozen-recipe aux residual CNN on the re-test folds (PLAN.md). Uses round4's own cnn_experiment.cnn_data and cnn_fitter.fit_one (imported
read-only, no bytecode written). Context on the training rows comes from this study's outer teachers. On V it comes from round4's strict-073 outer
predictions for partitions 1/2 (exactly as round4 did for partition 0), and from this study's 'full' teacher for the holdout.
usage: python -B cnn_retest.py WORKER NWORK   -> cnn/p<p>-fold<k>-seed<j>/ and cnn/p<p>-fold<k>-outer.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os, sys, json
from pathlib import Path
OUT = Path(__file__).resolve().parent; sys.dont_write_bytecode = True; os.environ['MPLCONFIGDIR'] = str(_repo_path(OUT, 'mpl-cache'))
R4 = _repo_path(OUT.parent, 'rmse-research-20261015/round4-full073'); sys.path.insert(0, str(R4))
import numpy as np, torch
from threadpoolctl import threadpool_limits
import cnn_experiment as e                  # its import of cnn_fitter gives e.c
c = e.c
SEED_SHIFTS = (0, 100000)

def teacher(p, k, name):
    return np.load(_repo_path(OUT, f'teachers/fold{k}-p{p}-{name}/predictions.npz'))

def run(worker, nwork):
    d = c.load_data(); labels = d.pop('y'); G = json.load(open(_repo_path(OUT, 'label-graph.json')))
    strict = np.load(_repo_path(R4.parent, 'scope.npz'))['strict_blend']; sridge = np.load(_repo_path(R4.parent, 'consistency-strict.npz'))['production']
    dev = list(d['dev']); jobs = [(g, j) for g in G for j in (0, 1)]
    for t, (g, j) in enumerate(jobs):
        if t % nwork != worker: continue
        p, k = g['partition'], g['fold']; T, V = np.array(g['T'], int), np.array(g['V'], int)
        import time
        while not all((_repo_path(OUT, f"teachers/fold{k}-p{p}-{n['name']}/predictions.npz")).exists() for n in g['nodes']): time.sleep(30)   # wait for this fold's teachers
        if (_repo_path(OUT, f'cnn/p{p}-fold{k}-seed{j}-outer.npz')).exists(): continue
        base = np.full(500, np.nan); gap = np.full(500, np.nan); sources = []
        for n in g['nodes']:
            if n['name'] == 'full': continue
            z = teacher(p, k, n['name']); base[z['query']] = z['base']; gap[z['query']] = z['tab'] - z['ridge']
            sources.append(dict(train=len(z['train']), query=len(z['query']), node=n['name']))
        assert np.isfinite(base[T]).all()
        if p in (1, 2):
            ix = np.array([dev.index(i) for i in V]); legacy = d['C73_cnn'][p, ix]; non021 = (strict[p, ix] - .16 * legacy) / .84
            failed = d['fail'][p, ix]; non021[failed] = strict[p, ix][failed]; qgap = d['C73_tab'][p, ix] - sridge[p, ix]
        else:
            z = teacher(p, k, 'full'); assert np.array_equal(z['query'], V); non021 = z['base']; qgap = z['tab'] - z['ridge']
            failed = np.nan_to_num(d['X'][V, list(d['keys']).index('n_obj')], nan=0) < 20   # full_branch.load's segmentation-failure rule
        # raw context exactly as cnn_experiment.raw_context: [teacher base, tab-ridge gap, noise, u_ph, log1p n_obj, coverage]
        tb = base.copy(); tb[V] = non021; gp = gap.copy(); gp[V] = qgap; keys = list(d['keys'])
        raw = np.c_[tb, gp, d['noise'], d['X'][:, keys.index('u_ph')], np.log1p(np.maximum(np.nan_to_num(d['X'][:, keys.index('n_obj')], nan=0), 0)), d['X'][:, keys.index('coverage')]]
        data = e.cnn_data(d, T, V, labels[T].copy(), base, sources, k, 'outer_refit', raw)
        seed = 2026101501 + 100 * k + SEED_SHIFTS[j]; path = _repo_path(OUT, f'cnn/p{p}-fold{k}-seed{j}')
        rv = e.fitted(data, seed, path, 'aux'); rv = np.array(rv, float); rv[failed] = 0.
        np.savez_compressed(_repo_path(OUT, f'cnn/p{p}-fold{k}-seed{j}-outer.npz'), V=V, residual=rv, failed=failed, non021=non021, seed=seed)
        print(json.dumps(dict(done=f'p{p} fold{k} seed{j}', worker=worker)), flush=True)

if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2): run(int(sys.argv[1]), int(sys.argv[2]))
