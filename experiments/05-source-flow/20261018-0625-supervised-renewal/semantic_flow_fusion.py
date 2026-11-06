"""Does learned grain assignment add to the actual leading flow challenger?"""
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
from threadpoolctl import threadpool_limits
from semantic_grains import Measurement, load


def main():
    start = time.monotonic()
    while not (_repo_path(HERE, 'candidate-flow46-v1/final-verification.json')).exists():
        if time.monotonic()-start > 7200:
            raise RuntimeError('Final inference did not finish; fusion readout not started.')
        time.sleep(5)
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(2026101931)
    d = np.load(_repo_path(HERE, 'data.npz')); dev = d['dev']; fold = d['fold']; y = d['y'][dev]
    tr = np.flatnonzero(fold != 0); q = np.flatnonzero(fold == 0)
    src = _repo_path(HERE, 'runs/semantic-pfn-f0-c160-v2/outer')
    z = np.load(_repo_io(src/'measurement-features.npz'))
    assert np.array_equal(z['train'], dev[tr]) and np.array_equal(z['query'], dev[q])
    learned = np.zeros((400, 21)); learned[tr] = z['train_features']; learned[q] = z['query_features']
    data = load(d); model = Measurement(data, torch.zeros(500, 104), dev[tr])
    with torch.no_grad():
        fixed = model.extra(dev).numpy()
    del model, data; gc.collect()
    z = np.load(_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz'))
    pos = {str(v): i for i, v in enumerate(z['ids'])}; P = z['X'][[pos[str(v)] for v in d['ids'][dev]]].astype(float)
    z = np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz')); assert np.array_equal(z['ids'], d['ids'])
    X = np.c_[P, z['flow'][dev]]
    old = np.load(_repo_path(HERE, 'domain-semantics/readout/oof.npz'))['H300_flow_only'][q]
    leading = np.load(_repo_path(HERE, 'flow-linear-check/oof.npz'))['combined_system'][q]
    out = _repo_path(HERE, 'semantic-flow-fusion-f0'); out.mkdir(exist_ok=True)
    spec = dict(scope='Conditional existing H300/physical representations; learned21 features fitted only on this outerT320. No outer-label coefficient fitting.',
                hypothesis='Test complementarity to the actual leading flow46 system, not only the weaker physical104 predictor.',
                control='Fixed untrained21 semantic measurements, with otherwise identical171-column native readout.',
                formula='flow46_full_system + .384*eligible*(Tab(P104+F46+semantic21)-Tab(P104+F46)); existing linear contribution unchanged.',
                native_readout='float64, local TabPFN3.5 n8, seeds0/100/200, same originalfold0 T320/Q80.',
                learned_model_sha256=hashlib.sha256((src/'model.pt').read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(spec, indent=2))
    sys.path.insert(0, str(_repo_path(ROOT, 'experiments/push-300'))); import harness
    predictions = {}; report = {}
    for name, extra in [('fixed21', fixed), ('learned21', learned)]:
        A = np.c_[X, extra]; ps = []
        for seed in [0, 100, 200]:
            path = out/f'{name}-seed{seed}.npz'
            if path.exists():
                z = np.load(_repo_io(path)); assert np.array_equal(z['train'], tr) and np.array_equal(z['query'], q); v = z['prediction']
            else:
                m = harness.tabpfn_new('v3.5', seed, 'cpu', n_est=8); m.fit(A[tr], y[tr]); v = m.predict(A[q]); del m; gc.collect()
                np.savez(_repo_io(path), train=tr, query=q, prediction=v)
            ps.append(v)
        tab = np.mean(ps, 0); system = leading+.384*(~d['failed'][q])*(tab-old)
        predictions[name+'_tab'] = tab; predictions[name+'_system'] = system
        report[name] = dict(tab_rmse=float(np.mean((tab-y[q])**2)**.5), system_rmse=float(np.mean((system-y[q])**2)**.5))
        print('ARM', name, json.dumps(report[name]), flush=True)
    report['leading_flow46_rmse'] = float(np.mean((leading-y[q])**2)**.5)
    report['base083_rmse'] = float(np.mean((d['base083'][q]-y[q])**2)**.5)
    np.savez(_repo_io(out/'predictions.npz'), ids=d['ids'][dev[q]], y=y[q], leading=leading, **predictions)
    (out/'results.json').write_text(json.dumps(report, indent=2)); print('DONE', json.dumps(report), flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
