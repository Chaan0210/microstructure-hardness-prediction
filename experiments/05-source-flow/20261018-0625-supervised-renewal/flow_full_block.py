"""Apply the promising continuous fields across all existing TabPFN members.

Both sides are freshly fitted with identical settings, row order and seeds.
Inherited global auxiliary features make this a conditional development test.
"""
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
from threadpoolctl import threadpool_limits


def main(a):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    wait_start = time.monotonic()
    while not (_repo_path(HERE, 'candidate-flow46-v1/final-verification.json')).exists() or not (_repo_path(HERE, 'flow-content-ablation/results.json')).exists():
        if time.monotonic()-wait_start > 7200:
            raise RuntimeError('Earlier inference or ablation has not released CPU resources; block experiment not started.')
        time.sleep(5)
    d = np.load(_repo_path(HERE, 'data.npz')); dev = d['dev']; ids = d['ids'][dev]; y = d['y'][dev]; fold = d['fold']
    f = np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz')); assert np.array_equal(f['ids'], d['ids'])
    flow = f['flow'][dev]; inputs = json.loads((_repo_path(HERE, 'flow-full-block-inputs.json')).read_text())
    out = _repo_path(HERE, 'flow-full-block'); out.mkdir(exist_ok=True)
    spec = dict(change='Append fixed H300 continuous46 fields to each of3banks x4Uvariants x2feature_sets; preserve noise12 gate.',
                fitting='Fresh paired controls and augmented heads, native local TabPFN3.5 n8, seeds0/100/200+fold, sameT320/Q80, float64, source row order identical.',
                system='083 + .384*eligible*(newfullTab-refittedfullTab) + .256*eligible*(flow46meanLinear-P104meanLinear)',
                weights='Fixed existing block masses, no fit to Public or these validation results.',
                scope='Conditional global auxiliary features; not complete end-to-end confirmation.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), inputs=inputs)
    (out/'recipe.json').write_text(json.dumps(spec, indent=2))
    sys.path.insert(0, str(_repo_path(ROOT, 'experiments/push-300'))); import harness
    features = {}; legacy = {}
    for row in inputs:
        name = f"{row['bank']}-{row['variant']}-{row['feature_set']}"
        z = np.load(_repo_path(ROOT, row['features'])); pos = {str(v): i for i, v in enumerate(z['ids'])}
        features[name] = z['X'][[pos[str(v)] for v in ids]].astype(float)
        z = np.load(_repo_path(ROOT, row['original_predictions'])); pos = {str(v): i for i, v in enumerate(z['original_ids'])}
        ix = [pos[str(v)] for v in ids]
        assert np.array_equal(z['original_fold'][ix], fold) and np.allclose(z['original_y'][ix], y)
        legacy[name] = z['original_g'][ix]
    lin = np.load(_repo_path(HERE, 'flow-linear-check/oof.npz')); records = []
    for k in a.folds:
        tr = np.flatnonzero(fold != k); q = np.flatnonzero(fold == k); preds = {}
        for row in inputs:
            name = f"{row['bank']}-{row['variant']}-{row['feature_set']}"; X = features[name]
            for arm, A in [('control', X), ('flow', np.c_[X, flow])]:
                ps = []
                for seed in [0, 100, 200]:
                    path = out/f'{name}-{arm}-f{k}-s{seed}.npz'
                    if path.exists():
                        z = np.load(_repo_io(path)); assert np.array_equal(z['train'], tr) and np.array_equal(z['query'], q)
                        v = z['prediction']
                    else:
                        start = time.monotonic(); m = harness.tabpfn_new('v3.5', seed+k, 'cpu', n_est=8)
                        m.fit(A[tr], y[tr]); v = m.predict(A[q]); del m; gc.collect()
                        np.savez(_repo_io(path), train=tr, query=q, prediction=v)
                        print('MEMBER', k, name, arm, seed, time.monotonic()-start, flush=True)
                    ps.append(v)
                preds[name+'-'+arm] = np.mean(ps, 0)
        block = {}
        for arm in ['control', 'flow', 'legacy']:
            pp = []
            for bank in ['038', '039', '044']:
                for variant in ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']:
                    name = f'{bank}-{variant}'
                    nb = legacy[name+'-nb'][q] if arm == 'legacy' else preds[name+'-nb-'+arm]
                    full = legacy[name+'-full'][q] if arm == 'legacy' else preds[name+'-full-'+arm]
                    pp.append(np.where(d['noise'][dev[q]] < 12, nb, full))
            block[arm] = np.mean(pp, 0)
        eligible = ~d['failed'][q]; base = d['base083'][q]
        tab_system = base+.384*eligible*(block['flow']-block['control'])
        system = tab_system+.256*eligible*(lin['flow'][q]-lin['control'][q])
        rm = lambda v: float(np.mean((v-y[q])**2)**.5)
        r = dict(fold=k, control_block_rmse=rm(block['control']), flow_block_rmse=rm(block['flow']),
                 tab_system_rmse=rm(tab_system), combined_system_rmse=rm(system),
                 single_bank_flow46_system_rmse=rm(lin['combined_system'][q]), base083_rmse=rm(base),
                 legacy_vs_refit_block_rms=float(np.mean((block['legacy']-block['control'])**2)**.5))
        np.savez(_repo_io(out/f'predictions-f{k}.npz'), ids=ids[q], y=y[q], train=tr, query=q, system=system,
                 tab_system=tab_system, base083=base, single_bank=lin['combined_system'][q], **block)
        (out/f'results-f{k}.json').write_text(json.dumps(r, indent=2)); records.append(r)
        print('FOLD', json.dumps(r), flush=True)
    if set(a.folds) == set(range(5)):
        zz = [np.load(_repo_io(out/f'predictions-f{k}.npz')) for k in range(5)]
        yt = np.concatenate([z['y'] for z in zz]); result = dict(folds=records, full_system_protocol_passed=False)
        for name in ['system', 'tab_system', 'base083', 'single_bank', 'control', 'flow']:
            v = np.concatenate([z[name] for z in zz]); result[name+'_rmse'] = float(np.mean((v-yt)**2)**.5)
        (out/'results.json').write_text(json.dumps(result, indent=2)); print('DONE', json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    with threadpool_limits(2):
        main(p.parse_args())
