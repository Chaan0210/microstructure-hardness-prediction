"""Fit/predict a frozen local candidate; no upload, downloads, or holdout scoring.

fit locks the recipe and development-only kernel selection before reading all
training labels. predict performs inference only with the fixed 50/25/25 blend.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import time
from pathlib import Path
ROOT = _RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve() != (_repo_path(ROOT, '.venv')).resolve():
    raise SystemExit('Use project .venv/bin/python -I candidate_pipeline.py')
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold
import cnn_controls as engine
from compact_regression import BalancedKernel
from candidate_features import extract_candidate

BASELINE = _repo_path(ROOT, 'artifacts/run-005/best_model.joblib')
PREVIOUS = _repo_path(ROOT, 'artifacts/submission-candidate-002.csv')
VALIDATION = _repo_path(ROOT, 'artifacts/frozen-validation-016/report.json')
PACKAGES = ('numpy', 'scipy', 'Pillow', 'scikit-learn', 'scikit-image', 'joblib', 'torch')
CORE_SOURCES = ('candidate_pipeline.py', 'candidate_features.py', 'latent_features.py',
                'grain_features.py', 'compact_regression.py', 'cnn_controls.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def local(value, artifacts=False):
    p = Path(value)
    if p.is_absolute():
        raise ValueError('Use project-relative paths')
    p = Path(os.path.abspath(_repo_path(ROOT, p)))
    allowed = (_repo_path(ROOT, 'artifacts')).resolve() if artifacts else ROOT.parent
    if not p.is_relative_to(ROOT) or not p.resolve().is_relative_to(allowed):
        raise ValueError('Path escapes allowed project directory')
    return p


def packages():
    return {p: importlib.metadata.version(p) for p in PACKAGES}


def sources(panel):
    names = [*CORE_SOURCES]
    if panel == 'run-014':
        names.append('phase_features.py')
    return {n: sha(_repo_path(ROOT, n)) for n in names}


def protected():
    expected = json.loads((_repo_path(ROOT, 'artifacts/run-005/report.json')).read_text())['model_sha256']
    values = {str(p.relative_to(ROOT)): sha(p) for p in (BASELINE, PREVIOUS)}
    if values[str(BASELINE.relative_to(ROOT))] != expected:
        raise ValueError('Original baseline model hash mismatch')
    return values


def assert_protected(snapshot):
    for name, expected in snapshot.items():
        if sha(_repo_path(ROOT, name)) != expected:
            raise RuntimeError(f'Protected artifact changed: {name}')


def read_csv(path, prefix, count, targets):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['ID', 'hardness']:
            raise ValueError('Unexpected CSV schema')
        rows = list(reader)
    ids = [r['ID'] for r in rows]
    if len(ids) != count or len(set(ids)) != count or any(not re.fullmatch(prefix+r'_\d{6}', s) for s in ids):
        raise ValueError('Invalid row count or IDs')
    y = np.array([float(r['hardness']) for r in rows]) if targets else None
    if y is not None and not np.isfinite(y).all():
        raise ValueError('Nonfinite targets')
    return ids, y


def rmse(y, prediction):
    if y.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Invalid predictions')
    return float(np.sqrt(np.mean((y-prediction)**2)))


def extract_one(ident, split, panel):
    path = _repo_path(ROOT, 'data', split, ident+'.png')
    if not path.resolve().is_relative_to((_repo_path(ROOT, 'data', split)).resolve()):
        raise ValueError('Image symlink escapes selected data directory')
    with Image.open(_repo_io(path)) as im:
        if im.mode != 'L' or im.size != (256, 256):
            raise ValueError(f'Invalid image schema: {ident}')
        raw = np.asarray(im, dtype=np.uint8).copy()
        small = np.asarray(im.resize((128, 128), Image.Resampling.BOX), dtype=np.float32)/255.
    values, names = extract_candidate(raw, panel)
    return values, names, small, sha(path)


def extract_all(ids, split, panel, names):
    records = []
    for first in range(0, len(ids), 40):
        records.extend(joblib.Parallel(n_jobs=2)(joblib.delayed(extract_one)(s, split, panel)
                                               for s in ids[first:first+40]))
        print(f'{split}: extracted {min(first+40, len(ids))}/{len(ids)}', flush=True)
    if any(r[1] != names for r in records):
        raise ValueError('Candidate feature schema mismatch')
    X = np.stack([r[0] for r in records])
    images = np.stack([r[2] for r in records])[:, None]
    if not np.isfinite(X).all() or not np.isfinite(images).all():
        raise ValueError('Nonfinite extracted data')
    return X, images, dict(zip(ids, [r[3] for r in records]))


def choose_kernel(panel):
    """Read only development targets/features and select the fixed nine-grid KRR."""
    split_path = _repo_path(ROOT, 'artifacts/run-001/split_manifest.json')
    split = json.loads(split_path.read_text())
    oof_path = _repo_path(ROOT, 'artifacts/run-005/development_oof.csv')
    with oof_path.open() as f:
        rows = list(csv.DictReader(f))
    by_id = {r['ID']: r for r in rows}
    ids = sorted(by_id)
    if (len(rows) != 400 or len(ids) != 400 or set(ids) != set(split['development_folds'])
            or set(ids) & set(split['holdout'])):
        raise ValueError('Invalid development-only IDs')
    y = np.array([float(by_id[s]['hardness']) for s in ids])
    if not np.isfinite(y).all():
        raise ValueError('Invalid development targets')
    groups = np.array([split['group_by_id'][s] for s in ids])
    directory = _repo_path(ROOT, 'artifacts', panel)
    report_path, cache_path = directory/'report.json', directory/'development_features.npz'
    report = json.loads(report_path.read_text())
    names, blocks = report['features'], report['blocks']
    with np.load(_repo_io(cache_path), allow_pickle=False) as cache:
        cache_ids = cache['ids'].tolist()
        if len(cache_ids) != 400 or len(set(cache_ids)) != 400 or set(cache_ids) != set(ids):
            raise ValueError('Panel cache must contain the 400 development rows only')
        index = {s: i for i, s in enumerate(cache_ids)}
        X = cache['X'][[index[s] for s in ids]]
    if (X.shape != (400, len(names)) or len(blocks) != len(names)
            or len(set(names)) != len(names) or not np.isfinite(X).all()):
        raise ValueError('Invalid compact panel schema')
    folds = list(GroupKFold(n_splits=3, shuffle=True, random_state=20261005).split(X, y, groups))
    scores = []
    for bandwidth in (.25, 1., 4.):
        for alpha in (.01, .1, 1.):
            pred = np.full(len(y), np.nan)
            for tr, va in folds:
                if set(groups[tr]) & set(groups[va]):
                    raise ValueError('Development inner-fold group overlap')
                model = BalancedKernel(blocks, bandwidth, alpha).fit(X[tr], y[tr])
                pred[va] = model.predict(X[va])
            scores.append({'bandwidth': bandwidth, 'alpha': alpha, 'development_inner_rmse': rmse(y, pred)})
    selected = min(scores, key=lambda s: s['development_inner_rmse'])
    evidence = {'seed': 20261005, 'n_splits': 3, 'n_development': 400,
                'selected': selected, 'candidates': scores,
                'development_folds': {ids[i]: k for k, (_, va) in enumerate(folds) for i in va},
                'split_manifest_sha256': sha(split_path), 'development_oof_sha256': sha(oof_path),
                'panel_cache_sha256': sha(cache_path), 'panel_report_sha256': sha(report_path),
                'interpretation': 'Select kernel settings only; this score is not final validation.'}
    return selected, names, blocks, ids, y, X, evidence


def fit_cnn(X, y, ids, output, device):
    rng = np.random.default_rng(20261003)
    torch.manual_seed(20261003)
    mu, sd, ym, ys = float(X.mean()), float(X.std()), float(y.mean()), float(y.std())
    if sd <= 0 or ys <= 0:
        raise ValueError('Degenerate CNN normalization')
    Xt, yt = (X-mu)/max(sd, 1e-6), (y-ym)/ys
    model = engine.TinyCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=.0001)
    avg, curves = None, []
    for epoch in range(80):
        model.train()
        order, loss_sum = rng.permutation(len(y)), 0.
        for first in range(0, len(y), 32):
            ix = order[first:first+32]
            xb = np.rot90(Xt[ix], int(rng.integers(4)), axes=(2, 3))
            if rng.random() < .5:
                xb = xb[:, :, :, ::-1]
            xb = torch.from_numpy(np.ascontiguousarray(xb)).to(device)
            target = torch.tensor(yt[ix], dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb), target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite CNN training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            loss_sum += float(loss.detach().cpu())*len(ix)
        scheduler.step()
        curves.append({'epoch': epoch+1, 'augmented_train_rmse': float(np.sqrt(loss_sum/len(y))*ys),
                       'lr_after_step': scheduler.get_last_lr()[0]})
        if epoch >= 75:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if avg is None:
                avg = {k: v/5 for k, v in state.items()}
            else:
                for k in avg:
                    avg[k] += state[k]/5
        if (epoch+1) % 10 == 0:
            print(json.dumps(curves[-1]), flush=True)
    engine.sync(device)
    torch.save({'state_dict': avg, 'normalization': {'x_mean': mu, 'x_std': sd,
                'y_mean': ym, 'y_std': ys}, 'train_ids': ids, 'architecture': 'TinyCNN',
                'image_size': 128, 'seed': 20261003, 'epochs': 80, 'average_last_epochs': 5}, _repo_io(output))
    del model, opt, avg
    if device == 'mps':
        torch.mps.empty_cache()
    return curves


def predict_cnn(X, checkpoint, device):
    saved = torch.load(_repo_io(checkpoint), map_location='cpu', weights_only=True)
    n = saved['normalization']
    if any(not np.isfinite(v) for v in n.values()) or n['x_std'] <= 0 or n['y_std'] <= 0:
        raise ValueError('Invalid saved CNN normalization')
    model = engine.TinyCNN().to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    result = []
    with torch.no_grad():
        for first in range(0, len(X), 32):
            xb = torch.from_numpy((X[first:first+32]-n['x_mean'])/max(n['x_std'], 1e-6)).to(device)
            result.extend((model(xb).cpu().numpy()*n['y_std']+n['y_mean']).tolist())
    engine.sync(device)
    return np.array(result)


def fit(args):
    out = local(args.output, artifacts=True)
    if out.exists():
        raise FileExistsError('Final model directory must be new')
    snapshot = protected()
    validation = json.loads(VALIDATION.read_text())
    if ({r['seed'] for r in validation.get('seeds', [])} != {20261011, 20261012}
            or len(validation['seeds']) != 2 or args.panel not in validation['feature_runs']
            or 'elapsed_seconds' not in validation):
        raise ValueError('Completed frozen-validation-016 with both prespecified seeds is required')
    selected, names, blocks, dev_ids, dev_y, dev_X, selection = choose_kernel(args.panel)
    decision = {'panel': args.panel, 'ensemble_weights': {'baseline': .5, 'cnn': .25, 'kernel': .25},
                'kernel': selected, 'kernel_selection': selection,
                'feature_names': names, 'blocks': blocks, 'baseline_protected': snapshot,
                'validation_report': str(VALIDATION.relative_to(ROOT)),
                'validation_report_sha256': sha(VALIDATION),
                'validation_interpretation': 'Development split robustness, not independent validation',
                'cnn': {'seed': 20261003, 'image_size': 128, 'epochs': 80, 'batch_size': 32,
                        'average_last_epochs': 5, 'lr': .002, 'weight_decay': .01,
                        'optimizer': 'AdamW', 'scheduler': 'CosineAnnealingLR T_max80 eta_min0.0001',
                        'augmentation': 'batch D4 rotations/reflections', 'clip_grad_norm': 5.},
                'fit_scope': 'All official train500 after this decision; no holdout scoring or test reads',
                'source_sha256': sources(args.panel), 'packages': packages(),
                'license_references': {'project_code': 'Original project implementation; no downloaded model code/weights',
                    'dependencies': {p: {'version': importlib.metadata.version(p),
                        'license_expression': importlib.metadata.metadata(p).get('License-Expression'),
                        'reference': 'Installed distribution METADATA and license files'} for p in PACKAGES}}}
    out.mkdir(parents=True, exist_ok=False)
    # The recipe and development-only parameter choice are durably frozen first.
    write(out/'decision.json', decision)
    started = time.perf_counter()
    ids, y = read_csv(_repo_path(ROOT, 'data/train.csv'), 'TRAIN', 500, True)
    index = {s: i for i, s in enumerate(ids)}
    if not set(dev_ids).issubset(index):
        raise ValueError('Development IDs missing from full training CSV')
    if not np.array_equal(y[[index[s] for s in dev_ids]], dev_y):
        raise ValueError('Full training CSV development labels changed')
    X, images, image_hashes = extract_all(ids, 'train', args.panel, names)
    np.testing.assert_allclose(X[[index[s] for s in dev_ids]], dev_X, rtol=1e-10, atol=1e-12,
                               err_msg='Re-extracted panel differs from frozen development cache')
    kernel = BalancedKernel(blocks, selected['bandwidth'], selected['alpha']).fit(X, y)
    joblib.dump({'model': kernel, 'feature_names': names, 'blocks': blocks,
                 'train_ids': ids}, _repo_io(out/'kernel.joblib'))
    # Existing baseline was already fitted on train500; preserve its exact bytes.
    shutil.copyfile(_repo_io(BASELINE), _repo_io(out/'baseline.joblib'))
    if sha(out/'baseline.joblib') != snapshot[str(BASELINE.relative_to(ROOT))]:
        raise ValueError('Baseline copy integrity mismatch')
    torch.set_num_threads(2)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    curves = fit_cnn(images, y, ids, out/'cnn.pt', device)
    write(out/'training-curve.json', curves)
    if sources(args.panel) != decision['source_sha256']:
        raise RuntimeError('Source changed during final fit')
    assert_protected(snapshot)
    report = {'panel': args.panel, 'n_train': 500, 'device': device,
              'decision_sha256': sha(out/'decision.json'),
              'source_sha256': decision['source_sha256'], 'packages': decision['packages'],
              'model_sha256': {n: sha(out/n) for n in ('baseline.joblib', 'kernel.joblib', 'cnn.pt')},
              'train_csv_sha256': sha(_repo_path(ROOT, 'data/train.csv')), 'training_image_sha256': image_hashes,
              'training_id_order': ids, 'feature_names': names, 'blocks': blocks,
              'elapsed_seconds': time.perf_counter()-started,
              'validation': 'No new validation or holdout scoring; see frozen development validation',
              'test_used': False, 'protected_artifacts': snapshot,
              'reproducibility': 'Fixed seeds and CPU-portable checkpoint; MPS bitwise repeatability not guaranteed'}
    write(out/'report.json', report)
    print(json.dumps({'model': str(out.relative_to(ROOT)), 'seconds': report['elapsed_seconds']}), flush=True)


def predict(args):
    directory, output = local(args.model, artifacts=True), local(args.output, artifacts=True)
    manifest_path = output.with_suffix('.manifest.json')
    components_path = output.with_suffix('.components.csv')
    if any(p.exists() for p in (output, manifest_path, components_path)):
        raise FileExistsError('Prediction outputs must all be new')
    snapshot = protected()
    report = json.loads((directory/'report.json').read_text())
    if sha(directory/'decision.json') != report['decision_sha256']:
        raise ValueError('Locked decision integrity mismatch')
    decision = json.loads((directory/'decision.json').read_text())
    if sources(report['panel']) != report['source_sha256'] or packages() != report['packages']:
        raise ValueError('Source or package versions changed since final fit')
    if snapshot != report['protected_artifacts']:
        raise ValueError('Protected originals changed since final fit')
    for name, digest in report['model_sha256'].items():
        if sha(directory/name) != digest:
            raise ValueError(f'Model integrity mismatch: {name}')
    # joblib files are trusted, locally produced and integrity-checked above.
    baseline = joblib.load(_repo_io(directory/'baseline.joblib'))
    kernel_bundle = joblib.load(_repo_io(directory/'kernel.joblib'))
    sample_path = _repo_path(ROOT, 'data/sample_submission.csv')
    ids, _ = read_csv(sample_path, 'TEST', 1000, False)
    cache_path = _repo_path(ROOT, 'artifacts/test-prediction-features.npz')
    with np.load(_repo_io(cache_path), allow_pickle=False) as cache:
        if cache['ids'].tolist() != ids or cache['names'].tolist() != baseline['feature_names']:
            raise ValueError('Baseline test cache ID order or feature schema mismatch')
        Xbase, cached_hashes = cache['X'], cache['hashes'].tolist()
    if Xbase.shape != (1000, len(baseline['feature_names'])) or not np.isfinite(Xbase).all():
        raise ValueError('Invalid baseline test features')
    names = report['feature_names']
    if kernel_bundle['feature_names'] != names or kernel_bundle['blocks'] != report['blocks']:
        raise ValueError('Kernel schema mismatch')
    started = time.perf_counter()
    X, images, image_hashes = extract_all(ids, 'test', report['panel'], names)
    if cached_hashes != [image_hashes[s] for s in ids]:
        raise ValueError('Test image bytes differ from baseline feature cache')
    # No estimator fit, tuning, clipping, or target access occurs in prediction.
    base_pred = sum(baseline['weights'][n]*m.predict(Xbase) for n, m in baseline['models'].items())
    kernel_pred = kernel_bundle['model'].predict(X)
    torch.set_num_threads(2)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    cnn_pred = predict_cnn(images, directory/'cnn.pt', device)
    pred = .5*base_pred+.25*cnn_pred+.25*kernel_pred
    values = {'baseline': base_pred, 'cnn': cnn_pred, 'kernel': kernel_pred, 'ensemble': pred}
    if any(p.shape != (1000,) or not np.isfinite(p).all() for p in values.values()):
        raise ValueError('Invalid component or ensemble predictions')
    if sources(report['panel']) != report['source_sha256']:
        raise RuntimeError('Source changed during prediction')
    assert_protected(snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    with components_path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', *values])
        writer.writerows([ident, *[p[i] for p in values.values()]] for i, ident in enumerate(ids))
    with output.open('x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'hardness'])
        writer.writerows(zip(ids, pred))
    write(manifest_path, {'model': str(directory.relative_to(ROOT)), 'panel': report['panel'],
          'model_report_sha256': sha(directory/'report.json'), 'model_sha256': report['model_sha256'],
          'decision_sha256': report['decision_sha256'], 'source_sha256': report['source_sha256'],
          'packages': report['packages'], 'sample_submission_sha256': sha(sample_path),
          'baseline_test_feature_cache_sha256': sha(cache_path), 'image_sha256': image_hashes,
          'submission_sha256': sha(output), 'components_sha256': sha(components_path),
          'rows': 1000, 'exact_sample_order': True, 'prediction_only': True,
          'ensemble_weights': decision['ensemble_weights'], 'clipping': False,
          'device': device, 'elapsed_seconds': time.perf_counter()-started,
          'protected_artifacts': snapshot,
          'component_summary': {n: {'min': float(p.min()), 'max': float(p.max()),
                                    'mean': float(p.mean()), 'std': float(p.std())} for n, p in values.items()}})
    assert_protected(snapshot)
    print(f'Prepared {output.relative_to(ROOT)}; no upload performed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('fit')
    p.add_argument('--panel', choices=['run-012', 'run-014'], required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=fit)
    p = commands.add_parser('predict')
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
