"""Full500 readouts for a frozen ensemble of source-trained flow encoders."""
from pathlib import Path
import os, sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[k] = '2'
for k in ['TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME']:
    os.environ[k] = str(HERE/'tmp')
os.environ['HF_HOME'] = str(HERE/'tmp/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
import argparse, csv, gc, hashlib, json, time, importlib.metadata, fcntl
import numpy as np
import torch, joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def settings(out):
    r = json.loads((out/'frozen-recipe.json').read_text())
    assert r['status'].startswith('frozen_') and r['field_count'] == 38
    assert r['head_source_sha256'] == sha(Path(__file__))
    assert r['control_training_sha256'] == sha(HERE/r['control_training_context'])
    assert r['baseline_sha256'] == sha(ROOT/r['baseline_csv'])
    return r


def feature(out, r, tag, split, ids):
    path = out/f'features/{tag}-{split}.npz'; meta = json.loads(path.with_name(f'{tag}-{split}-complete.json').read_text())
    assert meta['complete'] and meta['feature_sha256'] == sha(path)
    assert meta['recipe_sha256'] == sha(out/'frozen-recipe.json')
    z = np.load(path); assert np.array_equal(z['ids'], ids) and z['flow_keys'].tolist() == r['flow_keys']
    assert z['flow'].shape == (len(ids), 38) and np.isfinite(z['flow']).all()
    return z['flow'], meta


def fit(out, tag):
    r = settings(out); assert tag in r['encoder_tags']
    c = np.load(HERE/r['control_training_context']); ids = c['ids']; y = c['y']
    assert len(ids) == len(set(ids)) == 500
    label = {v['ID']: float(v['hardness']) for v in csv.DictReader((ROOT/'data/train.csv').open())}
    assert np.array_equal(y, [label[str(v)] for v in ids])
    f, meta = feature(out, r, tag, 'train', ids); X = np.c_[c['physical'], f]
    dest = out/'models'/tag; dest.mkdir(parents=True, exist_ok=True)
    contract = dict(recipe_sha256=sha(out/'frozen-recipe.json'), training_context_sha256=r['control_training_sha256'], feature_sha256=meta['feature_sha256'], encoder_model_sha256=meta['model_sha256'])
    if (dest/'fit-contract.json').exists():
        assert json.loads((dest/'fit-contract.json').read_text()) == contract
    else:
        save(dest/'fit-contract.json', contract)
    for kind in ['raw', 'log']:
        path = dest/f'{kind}.joblib'
        if not path.exists():
            model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=100))
            model.fit(X, y if kind == 'raw' else np.log(y)); joblib.dump(model, path)
    sys.path.insert(0, str(ROOT/'experiments/push-300')); import harness
    from tabpfn import TabPFNRegressor
    checks = []; start = time.monotonic()
    for seed in [0, 100, 200]:
        path = dest/f'seed{seed}.tabpfn_fit'
        if path.exists():
            continue
        model = harness.tabpfn_new('v3.5', seed, 'cpu', n_est=8); model.fit(X, y); model.save_fit_state(path)
        original = model.predict(X[:3]); del model; gc.collect()
        model = TabPFNRegressor.load_from_fit_state(path, device='cpu'); replay = model.predict(X[:3])
        error = float(abs(original-replay).max()); assert error < 1e-5
        checks.append(dict(seed=seed, max_difference=error, technical_probe_only=True)); del model; gc.collect()
        print('FITTED', tag, seed, flush=True)
    files = [*dest.glob('*.joblib'), *dest.glob('*.tabpfn_fit')]
    save(dest/'fit-complete.json', dict(complete=True, rows=500, training_ids=ids.tolist(), test_inputs_read=False,
         seconds=time.monotonic()-start, roundtrip_checks=checks, model_hashes={p.name: sha(p) for p in files},
         versions={k:importlib.metadata.version(k) for k in ['torch','tabpfn','numpy','scipy','scikit-learn']}, **contract))


def predict(out, replay):
    r = settings(out); assert r['control_predictions_sha256'] == sha(HERE/r['control_predictions'])
    controls = np.load(HERE/r['control_predictions'])
    sample = list(csv.DictReader((ROOT/'data/sample_submission.csv').open())); ids = np.array([v['ID'] for v in sample])
    assert len(ids) == len(set(ids)) == 1000 and np.array_equal(controls['ids'], ids)
    z = np.load(ROOT/r['physical_test']); assert np.array_equal(z['ids'], ids) and z['keys'].tolist() == r['physical_keys']
    P = z['X'].astype(float)
    sys.path.insert(0, str(ROOT/'experiments/push-300')); import harness
    harness.tabpfn_new('v3.5', 0, 'cpu', n_est=8)
    from tabpfn import TabPFNRegressor
    output = {}; start = time.monotonic()
    for tag in r['encoder_tags']:
        f, meta = feature(out, r, tag, 'test', ids); X = np.c_[P, f]; dest = out/'models'/tag
        trained = json.loads((dest/'fit-complete.json').read_text())
        assert trained['complete'] and trained['recipe_sha256'] == sha(out/'frozen-recipe.json')
        assert trained['encoder_model_sha256'] == meta['model_sha256']
        for name, digest in trained['model_hashes'].items():
            assert sha(dest/name) == digest
        values = []
        for seed in [0, 100, 200]:
            model = TabPFNRegressor.load_from_fit_state(dest/f'seed{seed}.tabpfn_fit', device='cpu')
            values.append(model.predict(X)); del model; gc.collect(); print('INFERRED', tag, seed, flush=True)
        output[tag+'_tab'] = np.mean(values, 0)
        output[tag+'_raw'] = joblib.load(dest/'raw.joblib').predict(X)
        output[tag+'_log'] = np.exp(joblib.load(dest/'log.joblib').predict(X))
    avg = {kind: np.mean([output[tag+'_'+kind] for tag in r['encoder_tags']], 0) for kind in ['tab','raw','log']}
    base = controls['base083']; eligible = controls['eligible']
    p = base+eligible*(.384*(avg['tab']-controls['control_tab'])+.128*(avg['raw']-controls['control_linear'])+.128*(avg['log']-controls['control_log']))
    assert np.isfinite(p).all() and np.array_equal(p[~eligible], base[~eligible])
    path = out/('candidate-source-flow38-replay.csv' if replay else 'candidate-source-flow38.csv')
    with path.open('w', newline='') as file:
        writer = csv.writer(file); writer.writerow(['ID','hardness']); writer.writerows(zip(ids,[repr(float(v)) for v in p]))
    result = dict(rows=1000, ids_unique=True, sample_order_correct=True, schema=['ID','hardness'], finite=True,
                  failure_fallback_unchanged=True, csv_sha256=sha(path), recipe_sha256=sha(out/'frozen-recipe.json'),
                  no_upload=True, no_final_selection_change=True, inference_only=True, seconds=time.monotonic()-start,
                  encoder_prediction_averaging='Equal mean of each encoder\'s three-seed Tab prediction and its raw/log predictions; no feature or parameter averaging.')
    if replay:
        old = np.load(out/'test-predictions.npz'); error = float(abs(old['prediction']-p).max()); assert error < 1e-5
        result.update(replay_max_difference=error, csv_byte_identical=sha(path) == sha(out/'candidate-source-flow38.csv'))
        save(out/'replay-verification.json', result)
    else:
        np.savez(out/'test-predictions.npz', ids=ids, prediction=p, base083=base, eligible=eligible, **output)
        save(out/'csv-verification.json', result)
    print('VERIFIED', json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['fit','predict','replay']); p.add_argument('--candidate', required=True); p.add_argument('--encoder-tag')
    a = p.parse_args(); out = Path(a.candidate).resolve(); assert out.is_relative_to(HERE)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    with (HERE/'tmp/heavy-learning.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        with threadpool_limits(2):
            fit(out, a.encoder_tag) if a.action == 'fit' else predict(out, a.action == 'replay')
