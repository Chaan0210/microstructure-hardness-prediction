"""Frozen recipe robustness on development rows only; never independent validation.

Every baseline/CNN/kernel component is refitted within each new outer train fold.
No old OOF predictions enter ensembles. Feature panels are frozen CLI inputs.
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
import sys
import time
from pathlib import Path
ROOT = _RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve() != (_repo_path(ROOT, '.venv')).resolve():
    raise SystemExit('Use project .venv/bin/python -I frozen_validation.py')
sys.path.insert(0, str(ROOT))
import numpy as np
import joblib
import torch
from PIL import Image
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
import cnn_controls as engine
from compact_regression import BalancedKernel

SPLIT_SEEDS = (20261011, 20261012)
CNN_SEED = 20261003
INNER_SEED = 20261005
WIDTHS = (.25, 1., 4.)
ALPHAS = (.01, .1, 1.)
CAVEAT = ('Development-only split robustness, not independent validation. All 400 rows '
          'informed historical feature/model/recipe choices. Kernel inner CV controls '
          'only current bandwidth/alpha choices. Exact/D4 groups do not verify source '
          'family independence. Conditional bootstrap omits fitting/selection uncertainty; '
          'the two partitions reuse the same observations. No holdout evaluation.')


def write(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(values):
    a = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def rmse(y, pred):
    if y.shape != pred.shape or not len(y) or not np.isfinite(pred).all():
        raise ValueError('Invalid prediction shape or values')
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def project_path(value):
    p = Path(value)
    if p.is_absolute():
        raise ValueError('Use a project-relative output path')
    p = Path(os.path.abspath(_repo_path(ROOT, p)))
    if not p.is_relative_to(ROOT) or not p.resolve().is_relative_to((_repo_path(ROOT, 'artifacts')).resolve()):
        raise ValueError('Output must remain inside project artifacts/')
    return p


def read_inputs(feature_runs):
    manifest_path = _repo_path(ROOT, 'artifacts/run-001/split_manifest.json')
    sp = json.loads(manifest_path.read_text())
    base_oof_path = _repo_path(ROOT, 'artifacts/run-005/development_oof.csv')
    with base_oof_path.open() as f:
        rows = list(csv.DictReader(f))
    by_id = {r['ID']: r for r in rows}
    ids = sorted(by_id)
    if len(rows) != 400 or len(ids) != 400 or set(ids) != set(sp['development_folds']):
        raise ValueError('Baseline OOF must contain the 400 unique development IDs')
    if set(ids) & set(sp['holdout']):
        raise ValueError('Development/holdout ID overlap')
    y = np.array([float(by_id[s]['hardness']) for s in ids], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite development targets')
    groups = np.array([sp['group_by_id'][s] for s in ids])
    # The mixed cache is used only for feature columns of the allowlisted IDs.
    # Never load its y array, and never retain or score holdout feature rows.
    base_cache = _repo_path(ROOT, 'artifacts/run-003/train_features.npz')
    with np.load(_repo_io(base_cache), allow_pickle=False) as cache:
        cache_ids = cache['ids'].tolist()
        if len(cache_ids) != len(set(cache_ids)):
            raise ValueError('Duplicate baseline cache IDs')
        index = {s: i for i, s in enumerate(cache_ids)}
        Xbase = cache['X'][[index[s] for s in ids]]
    if len(Xbase) != 400 or not np.isfinite(Xbase).all():
        raise ValueError('Invalid selected development baseline features')
    model_path = _repo_path(ROOT, 'artifacts/run-005/best_model.joblib')
    # Trusted local artifact; only cloned estimator recipes are fitted below.
    bundle = joblib.load(_repo_io(model_path))
    if Xbase.shape[1] != len(bundle['feature_names']):
        raise ValueError('Baseline feature schema width mismatch')
    recipes = {n: clone(m) for n, m in bundle['models'].items()}
    weights = dict(bundle['weights'])
    if set(recipes) != set(weights) or not np.isclose(sum(weights.values()), 1.):
        raise ValueError('Invalid baseline component weights')
    source_files = [Path(__file__), _repo_path(ROOT, 'cnn_controls.py'), _repo_path(ROOT, 'compact_regression.py')]
    provenance = {
        'source_sha256': {p.name: sha(p) for p in source_files},
        'split_manifest_sha256': sha(manifest_path),
        'baseline_development_oof_sha256': sha(base_oof_path),
        'baseline_model_sha256': sha(model_path),
        'baseline_feature_cache': str(base_cache.relative_to(ROOT)),
        'baseline_selected_development_X_sha256': array_sha(Xbase),
        'baseline_feature_names': bundle['feature_names'],
        'development_ids': ids, 'development_target_sha256': array_sha(y),
        'panels': {},
    }
    panels = {}
    for run in feature_runs:
        directory = _repo_path(ROOT, 'artifacts', run)
        cache_path, report_path = directory/'development_features.npz', directory/'report.json'
        report = json.loads(report_path.read_text())
        names = report.get('features', report.get('feature_names'))
        blocks = report.get('blocks')
        with np.load(_repo_io(cache_path), allow_pickle=False) as cache:
            panel_ids = cache['ids'].tolist()
            if len(panel_ids) != 400 or len(set(panel_ids)) != 400 or set(panel_ids) != set(ids):
                raise ValueError(f'{run}: feature cache must contain development rows only')
            index = {s: i for i, s in enumerate(panel_ids)}
            X = cache['X'][[index[s] for s in ids]]
        if (not isinstance(names, list) or not isinstance(blocks, list)
                or X.shape != (400, len(names)) or len(blocks) != len(names)
                or len(set(names)) != len(names) or not np.isfinite(X).all()):
            raise ValueError(f'{run}: invalid features/blocks schema')
        panels[run] = {'X': X, 'names': names, 'blocks': blocks}
        provenance['panels'][run] = {
            'cache_sha256': sha(cache_path), 'report_sha256': sha(report_path),
            'selected_X_sha256': array_sha(X), 'feature_names': names, 'blocks': blocks,
        }
    return ids, y, groups, Xbase, recipes, weights, panels, provenance


def load_development_images(ids):
    images, hashes = [], {}
    for ident in ids:
        path = _repo_path(ROOT, 'data/train', ident + '.png')
        if not path.resolve().is_relative_to((_repo_path(ROOT, 'data/train')).resolve()):
            raise ValueError('Image path escapes project training directory')
        with Image.open(_repo_io(path)) as im:
            if im.mode != 'L' or im.size != (256, 256):
                raise ValueError(f'Unexpected image schema: {ident}')
            images.append(np.asarray(im.resize((128, 128), Image.Resampling.BOX),
                                     dtype=np.float32) / 255.)
        hashes[ident] = sha(path)
    return np.stack(images)[:, None], hashes


def cnn_fold(X, y, tr, va, ids, fold, device, checkpoint):
    rng = np.random.default_rng(CNN_SEED + fold)
    torch.manual_seed(CNN_SEED + fold)
    mu, sd = float(X[tr].mean()), float(X[tr].std())
    ym, ys = float(y[tr].mean()), float(y[tr].std())
    if sd <= 0 or ys <= 0:
        raise ValueError('Degenerate CNN normalization')
    Xt, yt = (X[tr]-mu)/max(sd, 1e-6), (y[tr]-ym)/ys
    model = engine.TinyCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=.0001)
    avg, curve = None, []
    for epoch in range(80):
        model.train()
        order, loss_sum = rng.permutation(len(tr)), 0.
        for first in range(0, len(tr), 32):
            ix = order[first:first+32]
            xb = np.rot90(Xt[ix], int(rng.integers(4)), axes=(2, 3))
            if rng.random() < .5:
                xb = xb[:, :, :, ::-1]
            xb = torch.from_numpy(np.ascontiguousarray(xb)).to(device)
            target = torch.tensor(yt[ix], dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb), target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite CNN loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            loss_sum += float(loss.detach().cpu()) * len(ix)
        scheduler.step()
        curve.append({'epoch': epoch+1,
                      'augmented_train_rmse': float(np.sqrt(loss_sum/len(tr))*ys),
                      'lr_after_step': scheduler.get_last_lr()[0]})
        if epoch >= 75:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if avg is None:
                avg = {k: v/5 for k, v in state.items()}
            else:
                for k in avg:
                    avg[k] += state[k]/5
        if (epoch+1) % 20 == 0:
            print(json.dumps({'component': 'cnn', 'fold': fold, **curve[-1]}), flush=True)
    model.load_state_dict(avg)
    model.eval()
    def predict(indices):
        result = []
        with torch.no_grad():
            for first in range(0, len(indices), 32):
                xb = torch.from_numpy((X[indices[first:first+32]]-mu)/max(sd, 1e-6)).to(device)
                result.extend((model(xb).cpu().numpy()*ys+ym).tolist())
        return np.array(result)
    prediction, train_prediction = predict(va), predict(tr)
    engine.sync(device)
    torch.save({'state_dict': avg, 'normalization': {'x_mean': mu, 'x_std': sd,
                'y_mean': ym, 'y_std': ys}, 'fold': fold, 'seed': CNN_SEED+fold,
                'architecture': 'cnn_controls.TinyCNN', 'image_size': 128,
                'epochs': 80, 'average_last_epochs': 5,
                'train_ids': [ids[i] for i in tr], 'validation_ids': [ids[i] for i in va]}, _repo_io(checkpoint))
    summary = {'train_rmse': rmse(y[tr], train_prediction),
               'validation_rmse': rmse(y[va], prediction), 'curve': curve}
    del model, opt, avg
    if device == 'mps':
        torch.mps.empty_cache()
    return prediction, summary


def kernel_fold(X, y, groups, blocks, tr, va, fold):
    inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=INNER_SEED+fold)
                 .split(X[tr], y[tr], groups[tr]))
    scores = []
    for width in WIDTHS:
        for alpha in ALPHAS:
            pred = np.full(len(tr), np.nan)
            for ti, vi in inner:
                if set(groups[tr[ti]]) & set(groups[tr[vi]]):
                    raise ValueError('Inner group overlap')
                model = BalancedKernel(blocks, width, alpha).fit(X[tr[ti]], y[tr[ti]])
                pred[vi] = model.predict(X[tr[vi]])
            scores.append({'bandwidth': width, 'alpha': alpha, 'inner_rmse': rmse(y[tr], pred)})
    chosen = min(scores, key=lambda z: z['inner_rmse'])
    model = BalancedKernel(blocks, chosen['bandwidth'], chosen['alpha']).fit(X[tr], y[tr])
    return model.predict(X[va]), model, {'selected': chosen, 'inner_scores': scores}


def metrics(y, predictions, fid, seed):
    rng = np.random.default_rng(20261020 + seed)
    draws = rng.integers(0, len(y), (5000, len(y)))
    base_sq = (y-predictions['baseline'])**2
    result = {}
    for name, pred in predictions.items():
        score = rmse(y, pred)
        delta = np.sqrt(base_sq[draws].mean(axis=1))-np.sqrt(((y-pred)**2)[draws].mean(axis=1))
        fold_scores = [rmse(y[fid == k], pred[fid == k]) for k in range(5)]
        base_folds = [rmse(y[fid == k], predictions['baseline'][fid == k]) for k in range(5)]
        result[name] = {'oof_rmse': score, 'fold_rmse': fold_scores,
                        'improvement_vs_baseline': rmse(y, predictions['baseline'])-score,
                        'fold_improvement_vs_baseline': [a-b for a, b in zip(base_folds, fold_scores)],
                        'conditional_paired_improvement_95': np.percentile(delta, [2.5, 97.5]).tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-runs', nargs='+', required=True, choices=['run-012', 'run-014'])
    parser.add_argument('--output', required=True, help='Unique project-relative artifacts/ directory')
    args = parser.parse_args()
    if not 1 <= len(args.feature_runs) <= 2 or len(set(args.feature_runs)) != len(args.feature_runs):
        parser.error('Choose one or two distinct frozen feature panels before running')
    out = project_path(args.output)
    if out.exists():
        raise FileExistsError('Output must be unique; existing runs are never overwritten')
    ids, y, groups, Xbase, recipes, weights, panels, provenance = read_inputs(args.feature_runs)
    torch.set_num_threads(2)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    plan = {'feature_runs': args.feature_runs, 'outer_split_seeds': list(SPLIT_SEEDS),
            'outer_folds': 5, 'inner_folds': 3, 'inner_seed_plus_fold': INNER_SEED,
            'kernel_bandwidths': list(WIDTHS), 'kernel_alphas': list(ALPHAS),
            'cnn': {'architecture': 'TinyCNN', 'seed_plus_fold': CNN_SEED,
                    'image_size': 128, 'epochs': 80, 'average_last_epochs': 5,
                    'batch_size': 32, 'optimizer': 'AdamW', 'lr': .002, 'weight_decay': .01,
                    'schedule': 'CosineAnnealingLR(T_max=80, eta_min=0.0001)',
                    'augmentation': 'batch D4', 'endpoint': 'fixed; no early stopping'},
            'baseline_component_weights': weights,
            'ensemble_weights': {'baseline': .5, 'cnn': .25, 'nested_kernel': .25},
            'panel_selection': 'Frozen CLI choices; report every panel, no winner selection',
            'old_oof_prediction_reuse': False, 'n_development': 400,
            'device': device, 'packages': {p: importlib.metadata.version(p) for p in
                       ('torch', 'numpy', 'scipy', 'Pillow', 'scikit-learn', 'joblib')},
            'caveat': CAVEAT}
    out.mkdir(parents=True, exist_ok=False)
    write(out/'plan.json', plan)
    started = time.perf_counter()
    Ximage, image_hashes = load_development_images(ids)
    provenance['development_image_sha256'] = image_hashes
    write(out/'input-provenance.json', provenance)
    seed_reports = []
    for seed in SPLIT_SEEDS:
        directory = out/f'seed-{seed}'
        directory.mkdir()
        folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(Xbase, y, groups))
        fid = np.full(len(y), -1, dtype=int)
        predictions = {n: np.full(len(y), np.nan) for n in ['baseline', 'cnn',
                         *[f'kernel_{p}' for p in panels], *[f'ensemble_{p}' for p in panels]]}
        for fold, (_, va) in enumerate(folds):
            fid[va] = fold
        if np.any(fid < 0):
            raise ValueError('Incomplete outer folds')
        write(directory/'split_manifest.json', {'seed': seed,
              'development_folds': dict(zip(ids, fid.tolist())),
              'development_groups': dict(zip(ids, groups.tolist()))})
        fold_reports = []
        for fold, (tr, va) in enumerate(folds):
            if set(groups[tr]) & set(groups[va]):
                raise ValueError('Outer group overlap')
            fold_start = time.perf_counter()
            checkpoint_dir = directory/f'fold-{fold}'
            checkpoint_dir.mkdir()
            t = time.perf_counter()
            fitted = {n: clone(m).fit(Xbase[tr], y[tr]) for n, m in recipes.items()}
            predictions['baseline'][va] = sum(weights[n]*m.predict(Xbase[va]) for n, m in fitted.items())
            joblib.dump({'models': fitted, 'weights': weights,
                         'feature_names': provenance['baseline_feature_names'],
                         'train_ids': [ids[i] for i in tr]}, _repo_io(checkpoint_dir/'baseline.joblib'))
            baseline_seconds = time.perf_counter()-t
            del fitted
            t = time.perf_counter()
            predictions['cnn'][va], cnn_summary = cnn_fold(Ximage, y, tr, va, ids, fold, device,
                                                          checkpoint_dir/'cnn.pt')
            cnn_seconds = time.perf_counter()-t
            kernels = {}
            for panel, data in panels.items():
                t = time.perf_counter()
                pred, model, selection = kernel_fold(data['X'], y, groups, data['blocks'], tr, va, fold)
                predictions[f'kernel_{panel}'][va] = pred
                predictions[f'ensemble_{panel}'][va] = (.5*predictions['baseline'][va]
                                                       + .25*predictions['cnn'][va] + .25*pred)
                joblib.dump({'model': model, 'feature_names': data['names'], 'blocks': data['blocks'],
                             'train_ids': [ids[i] for i in tr]}, _repo_io(checkpoint_dir/f'kernel-{panel}.joblib'))
                kernels[panel] = {**selection, 'seconds': time.perf_counter()-t,
                                  'validation_rmse': rmse(y[va], pred)}
            record = {'fold': fold, 'n_train': len(tr), 'n_validation': len(va),
                      'baseline_seconds': baseline_seconds, 'cnn_seconds': cnn_seconds,
                      'baseline_validation_rmse': rmse(y[va], predictions['baseline'][va]),
                      'cnn': cnn_summary, 'kernels': kernels,
                      'elapsed_seconds': time.perf_counter()-fold_start,
                      'checkpoint_sha256': {p.name: sha(p) for p in checkpoint_dir.iterdir()}}
            write(directory/f'fold-{fold}-report.json', record)
            fold_reports.append(record)
            print(json.dumps({'split_seed': seed, 'fold': fold, 'seconds': record['elapsed_seconds'],
                              'baseline_rmse': record['baseline_validation_rmse'],
                              'cnn_rmse': cnn_summary['validation_rmse'],
                              'ensemble_rmse': {p: rmse(y[va], predictions[f'ensemble_{p}'][va])
                                                for p in panels}}), flush=True)
        scores = metrics(y, predictions, fid, seed)
        with (directory/'development_oof.csv').open('x', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ID', 'hardness', 'fold', *predictions])
            w.writerows([ident, y[i], fid[i], *[p[i] for p in predictions.values()]]
                        for i, ident in enumerate(ids))
        report = {'seed': seed, 'scores': scores, 'folds': fold_reports, 'caveat': CAVEAT}
        write(directory/'report.json', report)
        seed_reports.append({'seed': seed, 'scores': scores})
    for name, digest in provenance['source_sha256'].items():
        if sha(_repo_path(ROOT, name)) != digest:
            raise RuntimeError('Source changed during validation; inspect partial outputs')
    write(out/'report.json', {**plan, 'seeds': seed_reports,
                             'elapsed_seconds': time.perf_counter()-started,
                             'aggregation': 'No pooled 800-row CI: partitions share the same 400 labels'})
    print(json.dumps({'output': str(out.relative_to(ROOT)), 'seeds': seed_reports}), flush=True)


if __name__ == '__main__':
    main()
