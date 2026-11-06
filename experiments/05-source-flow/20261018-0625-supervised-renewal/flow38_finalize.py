"""Frozen38-field challenger; full500 fitting and inference-only cache reuse."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, sys
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
CANONICAL = _repo_path(HERE, 'candidate-flow38-v1')
sys.dont_write_bytecode = True
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[k] = '2'
for k in ['TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME']:
    os.environ[k] = str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME'] = str(_repo_path(HERE, 'tmp/huggingface'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
import argparse, csv, gc, hashlib, json, shutil, time, importlib.metadata
import numpy as np
import torch, joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False))


def recipe(out):
    r = json.loads((out/'frozen-recipe.json').read_text())
    assert r['status'].startswith('frozen_') and r['projection_indices'] == list(range(38))
    assert sha(Path(__file__)) == r['implementation_sha256']
    assert sha(_repo_path(ROOT, r['baseline_csv'])) == r['baseline_sha256']
    assert sha(_repo_path(HERE, r['training_context'])) == r['training_context_sha256']
    return r


def fit(out):
    r = recipe(out); c = np.load(_repo_path(HERE, r['training_context']))
    ids, y = c['ids'], c['y']; X = np.c_[c['physical'], c['flow'][:, r['projection_indices']]]
    assert len(ids) == len(set(ids)) == 500 and X.shape == (500, 142)
    labels = {v['ID']: float(v['hardness']) for v in csv.DictReader((_repo_path(ROOT, 'data/train.csv')).open())}
    assert np.array_equal(y, [labels[str(v)] for v in ids])
    contract = dict(recipe_sha256=sha(out/'frozen-recipe.json'), training_context_sha256=r['training_context_sha256'])
    if (out/'fit-contract.json').exists():
        assert json.loads((out/'fit-contract.json').read_text()) == contract
    else:
        save(out/'fit-contract.json', contract)
    models = out/'models'; models.mkdir(exist_ok=True)
    for kind in ['raw', 'log']:
        p = models/f'flow38-{kind}.joblib'
        if not p.exists():
            m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=100))
            m.fit(X, y if kind == 'raw' else np.log(y)); joblib.dump(m, _repo_io(p))
    sys.path.insert(0, str(_repo_path(ROOT, 'experiments/push-300'))); import harness
    from tabpfn import TabPFNRegressor
    checks = []; start = time.monotonic()
    for seed in [0, 100, 200]:
        p = models/f'flow38-seed{seed}.tabpfn_fit'
        if p.exists():
            continue
        m = harness.tabpfn_new('v3.5', seed, 'cpu', n_est=8); m.fit(X, y); m.save_fit_state(p)
        expected = m.predict(X[:3]); del m; gc.collect()
        m = TabPFNRegressor.load_from_fit_state(p, device='cpu'); actual = m.predict(X[:3]); error = float(abs(expected-actual).max())
        assert error < 1e-5; checks.append(dict(seed=seed, saved_state_roundtrip_max_difference=error, technical_only=True))
        del m; gc.collect(); print('FITTED', seed, flush=True)
    save(out/'fit-complete.json', dict(complete=True, original_training_rows=500, training_ids=ids.tolist(),
         test_inputs_read=False, model_hashes={str(p.relative_to(out)): sha(p) for p in models.iterdir()},
         roundtrip_checks=checks, seconds=time.monotonic()-start, **contract,
         versions={k: importlib.metadata.version(k) for k in ['torch', 'tabpfn', 'numpy', 'scipy', 'scikit-learn']}))
    print('FIT_COMPLETE', flush=True)


def predict(out, replay):
    r = recipe(out); fitted = json.loads((out/'fit-complete.json').read_text())
    assert fitted['recipe_sha256'] == sha(out/'frozen-recipe.json')
    for name, digest in fitted['model_hashes'].items():
        assert sha(out/name) == digest
    assert sha(_repo_path(HERE, r['control_predictions'])) == r['control_predictions_sha256']
    control = np.load(_repo_path(HERE, r['control_predictions']))
    sample = list(csv.DictReader((_repo_path(ROOT, 'data/sample_submission.csv')).open())); ids = np.array([v['ID'] for v in sample])
    assert len(ids) == len(set(ids)) == 1000 and np.array_equal(control['ids'], ids)
    p = np.load(_repo_path(ROOT, r['physical_test'])); f = np.load(_repo_path(HERE, r['test_flow_cache']))
    assert np.array_equal(p['ids'], ids) and np.array_equal(f['ids'], ids)
    assert p['keys'].tolist() == r['physical_keys'] and f['flow_keys'].tolist() == r['source_flow_keys']
    F = f['flow'][:, r['projection_indices']]; assert F.shape == (1000, 38) and np.isfinite(F).all()
    X = np.c_[p['X'].astype(float), F]
    base = np.array([float(v['hardness']) for v in csv.DictReader((_repo_path(ROOT, r['baseline_csv'])).open())])
    assert np.array_equal(base, control['base083'])
    eligible = control['eligible']; assert eligible.dtype == np.bool_
    sys.path.insert(0, str(_repo_path(ROOT, 'experiments/push-300'))); import harness
    harness.tabpfn_new('v3.5', 0, 'cpu', n_est=8)
    from tabpfn import TabPFNRegressor
    ps = []; start = time.monotonic()
    for seed in [0, 100, 200]:
        m = TabPFNRegressor.load_from_fit_state(out/f'models/flow38-seed{seed}.tabpfn_fit', device='cpu')
        ps.append(m.predict(X)); del m; gc.collect(); print('INFERRED', seed, flush=True)
    tab = np.mean(ps, 0); raw = joblib.load(_repo_io(out/'models/flow38-raw.joblib')).predict(X)
    log = np.exp(joblib.load(_repo_io(out/'models/flow38-log.joblib')).predict(X))
    prediction = base+eligible*(.384*(tab-control['control_tab'])+.128*(raw-control['control_linear'])+.128*(log-control['control_log']))
    assert prediction.shape == (1000,) and np.isfinite(prediction).all()
    assert np.array_equal(prediction[~eligible], base[~eligible])
    target = out/('candidate-flow38-v1-replay.csv' if replay else 'candidate-flow38-v1.csv')
    with target.open('w', newline='') as file:
        writer = csv.writer(file); writer.writerow(['ID', 'hardness']); writer.writerows(zip(ids, [repr(float(v)) for v in prediction]))
    report = dict(rows=1000, ids_unique=True, ids_match_sample_order=True, schema=['ID', 'hardness'], finite_predictions=True,
                  frozen_failure_fallback_unchanged=True, csv_sha256=sha(target), recipe_sha256=sha(out/'frozen-recipe.json'),
                  inference_only=True, no_upload=True, no_selection_change=True, seconds=time.monotonic()-start,
                  reused_controls='Previously verified identical full500physical104 models, seeds0/100/200; pinned training context and inference cache.')
    if replay:
        previous = np.load(_repo_io(out/'test-predictions.npz')); error = float(abs(previous['prediction']-prediction).max()); assert error < 1e-5
        report.update(replay_max_difference=error, csv_byte_identical=sha(target) == sha(out/'candidate-flow38-v1.csv'))
        save(out/'replay-verification.json', report)
    else:
        np.savez(_repo_io(out/'test-predictions.npz'), ids=ids, prediction=prediction, base083=base, eligible=eligible, flow38_tab=tab, flow38_raw=raw, flow38_log=log)
        save(out/'csv-verification.json', report)
    print('CSV_VERIFIED', json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['fit', 'predict', 'replay']); p.add_argument('--output-dir')
    a = p.parse_args(); out = Path(a.output_dir).resolve() if a.output_dir else CANONICAL
    assert out.is_relative_to(HERE) and out != HERE
    out.mkdir(parents=True, exist_ok=True)
    if not (out/'frozen-recipe.json').exists():
        assert a.action == 'fit' and out != CANONICAL
        shutil.copyfile(_repo_path(CANONICAL, 'frozen-recipe.json'), out/'frozen-recipe.json')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    with threadpool_limits(2):
        fit(out) if a.action == 'fit' else predict(out, a.action == 'replay')
