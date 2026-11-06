"""Separate learned continuous geometry from image-local photometry."""
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
import gc, hashlib, json, time
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def main():
    start = time.monotonic()
    while not (_repo_path(HERE, 'flow-geometry/readout/results-f1.json')).exists() or not (_repo_path(HERE, 'source-h300/repeat0-fold0-seed1811/component-results.json')).exists():
        if time.monotonic()-start > 7200:
            raise RuntimeError('Earlier readout did not finish; no ablation started.')
        time.sleep(5)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    d = np.load(_repo_path(HERE, 'data.npz')); dev = d['dev']; y = d['y'][dev]; fold = d['fold']
    z = np.load(_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz')); pos = {str(v): i for i, v in enumerate(z['ids'])}
    P = z['X'][[pos[str(v)] for v in d['ids'][dev]]].astype(float)
    z = np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz')); assert np.array_equal(z['ids'], d['ids'])
    F = z['flow'][dev]; keys = z['flow_keys'].tolist()
    assert keys[38:] == ['gray_moment1', 'gray_moment2', 'gray_moment3', 'gray_moment4', 'brightness_bin0', 'brightness_bin1', 'brightness_bin2', 'brightness_bin3']
    out = _repo_path(HERE, 'flow-content-ablation'); out.mkdir(exist_ok=True)
    spec = dict(arms={'photometry8': keys[38:], 'learned_fields38': keys[:38]},
                hypothesis='Check whether the gain requires learned flow structure or only fixed image-local brightness statistics.',
                method='Fixed native TabPFN3.5 n8x3 seeds0/100/200+fold, Ridge100 raw/log, complete083 paired additions at inherited .384/.128/.128 masses.',
                scope='Historical physical104 remains conditional. Photometry8 has no learned image representation; learned_fields38 inherits global H300 exposure.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(spec, indent=2))
    sys.path.insert(0, str(_repo_path(ROOT, 'experiments/push-300'))); import harness
    oldtab = np.load(_repo_path(HERE, 'domain-semantics/readout/oof.npz'))['physical']; oldlin = np.load(_repo_path(HERE, 'flow-linear-check/oof.npz'))
    pred = {}
    for name, extra in [('photometry8', F[:, 38:]), ('learned_fields38', F[:, :38])]:
        X = np.c_[P, extra]; tab = np.zeros(400); lin = np.zeros(400)
        for k in range(5):
            tr = np.flatnonzero(fold != k); q = np.flatnonzero(fold == k); ps = []
            for seed in [0, 100, 200]:
                path = out/f'{name}-f{k}-s{seed}.npz'
                if path.exists():
                    z = np.load(_repo_io(path)); assert np.array_equal(z['train'], tr) and np.array_equal(z['query'], q); v = z['prediction']
                else:
                    m = harness.tabpfn_new('v3.5', seed+k, 'cpu', n_est=8); m.fit(X[tr], y[tr]); v = m.predict(X[q]); del m; gc.collect()
                    np.savez(_repo_io(path), train=tr, query=q, prediction=v)
                ps.append(v)
            tab[q] = np.mean(ps, 0); ls = []
            for kind in ['raw', 'log']:
                m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=100))
                m.fit(X[tr], y[tr] if kind == 'raw' else np.log(y[tr])); v = m.predict(X[q]); ls.append(v if kind == 'raw' else np.exp(v))
            lin[q] = np.mean(ls, 0)
            print('FOLD', name, k, flush=True)
        pred[name+'_tab'] = tab; pred[name+'_linear'] = lin
        pred[name+'_system'] = d['base083']+(~d['failed'])*(.384*(tab-oldtab)+.256*(lin-oldlin['control']))
        print('ARM', name, float(np.mean((pred[name+'_system']-y)**2)**.5), flush=True)
    pred['flow46_system'] = oldlin['combined_system']; pred['base083'] = d['base083']
    result = {k: dict(rmse=float(np.mean((v-y)**2)**.5), fold_rmse=[float(np.mean((v[fold == f]-y[fold == f])**2)**.5) for f in range(5)]) for k, v in pred.items()}
    np.savez(_repo_io(out/'oof.npz'), ids=d['ids'][dev], y=y, fold=fold, **pred)
    (out/'results.json').write_text(json.dumps(result, indent=2)); print('DONE', json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
