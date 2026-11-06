"""Shared read-only legacy loading and evaluation for isolated study900.

All explicit writes are confined to this new study directory.
Historical representations and the reused development sample make results conditional.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
PARTS = ('original', '20261031', '20261101')
VARIANTS = ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def destination(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE):
        raise ValueError(f'write outside isolated study: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path, data):
    destination(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def prepare_base():
    e = _repo_path(ROOT, 'experiments')
    sets = {}
    sources = {}
    for c, folder, tag in [('038', 'push-300', 'v35'), ('039', 'push-600', 'angh'), ('044', 'push-800', 'anghcor')]:
        sets[c] = []
        for v in VARIANTS:
            path = e / folder / f'dev-{v}-{tag}.npz'
            sources[str(path.relative_to(ROOT))] = sha(path)
            with np.load(_repo_io(path), allow_pickle=False) as d:
                sets[c].append({k: d[k] for k in d.files})
    ref = sets['038'][0]['original_ids'].astype(str)
    pmat, gmat, bmat, fs, folds, ys = [], [], [], [], [], []
    for part in PARTS:
        cg = []
        part_y = part_p = part_f = part_fold = None
        for c in sets:
            vg = []
            for d in sets[c]:
                ids = d[f'{part}_ids'].astype(str)
                assert len(set(ids)) == len(ref) and set(ids) == set(ref)
                lookup = {i: j for j, i in enumerate(ids)}
                ix = np.array([lookup[i] for i in ref])
                y, p, failed, fold = [d[f'{part}_{k}'][ix] for k in ('y', 'p021', 'failed', 'fold')]
                if part_y is not None:
                    for a, b in [(y, part_y), (p, part_p), (failed, part_f), (fold, part_fold)]:
                        np.testing.assert_array_equal(a, b)
                part_y, part_p, part_f, part_fold = y, p, failed, fold
                vg.append(d[f'{part}_g'][ix])
            cg.append(np.mean(vg, axis=0))
        g = np.where(part_f, part_p, np.mean(cg, axis=0))
        pmat.append(part_p); gmat.append(g); bmat.append(.2 * part_p + .8 * g)
        fs.append(part_f); folds.append(part_fold); ys.append(part_y)
    for y in ys:
        np.testing.assert_array_equal(y, ys[0])
    fpath = e / 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'
    sources[str(fpath.relative_to(ROOT))] = sha(fpath)
    with np.load(_repo_io(fpath), allow_pickle=False) as d:
        pos = {str(i): j for j, i in enumerate(d['ids'])}
        X = d['X'][[pos[i] for i in ref]].astype(float)
        keys = d['keys'].astype(str)
    labels = {r['ID']: float(r['hardness']) for r in csv.DictReader((_repo_path(ROOT, 'data/train.csv')).open())}
    np.testing.assert_allclose(ys[0], [labels[i] for i in ref], atol=0, rtol=0)
    out = dict(ids=ref, y=ys[0], fold=np.array(folds), p021=np.array(pmat), grain=np.array(gmat), baseline=np.array(bmat), failed=np.array(fs),
               X98=X, keys98=keys, noise=255 * X[:, list(keys).index('noise')], Nproxy=65536 / np.exp(X[:, list(keys).index('m_la_mean')]), split_names=np.array(PARTS))
    assert out['baseline'].shape == (3, 400)
    rmse = float(np.sqrt(np.mean((out['baseline'] - out['y']) ** 2)))
    assert abs(rmse - 9.940240144126825) < 1e-10
    np.savez(_repo_io(destination(_repo_path(HERE, 'base.npz'))), **out)
    dump_json(_repo_path(HERE, 'base-verification.json'), dict(rmse=rmse, sources=sources, row_alignment=True, exact045_formula=True,
              unique_images=400, repetitions=3, note='Same 400 repeatedly developed images, not 1200 independent observations.'))
    return out


def load_base():
    with np.load(_repo_path(HERE, 'base.npz'), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def evaluate(pred, base=None, bootstrap=5000):
    d = load_base() if base is None else base
    p = np.asarray(pred, dtype=float)
    if p.shape != d['baseline'].shape or not np.isfinite(p).all():
        raise ValueError('Expected finite prediction[3,400]')
    se = (p - d['y']) ** 2; bse = (d['baseline'] - d['y']) ** 2
    avg, old = se.mean(0), bse.mean(0)
    q = np.nanquantile(d['noise'], [1/3, 2/3])
    groups = {'all': np.ones(400, bool), 'clean': d['noise'] < q[0], 'middle': (d['noise'] >= q[0]) & (d['noise'] < q[1]),
              'noisy': d['noise'] >= q[1], 'noisy_coarse_medium': (d['noise'] >= q[1]) & (d['Nproxy'] < 300)}
    r = {'rmse': float(np.sqrt(avg.mean())), 'baseline_rmse': float(np.sqrt(old.mean())),
         'rmse_gain': float(np.sqrt(old.mean()) - np.sqrt(avg.mean())), 'mse_gain': float((old - avg).mean()),
         'partitions': {}, 'groups': {}, 'error_correlation': float(np.corrcoef((p-d['y']).ravel(), (d['baseline']-d['y']).ravel())[0,1])}
    for n, part in enumerate(d['split_names']):
        gain = [float(np.sqrt(bse[n, d['fold'][n] == k].mean()) - np.sqrt(se[n, d['fold'][n] == k].mean())) for k in range(5)]
        r['partitions'][str(part)] = {'rmse': float(np.sqrt(se[n].mean())), 'gain': float(np.sqrt(bse[n].mean())-np.sqrt(se[n].mean())), 'fold_gain': gain}
    for name, mask in groups.items():
        r['groups'][name] = {'n': int(mask.sum()), 'rmse': float(np.sqrt(avg[mask].mean())), 'baseline': float(np.sqrt(old[mask].mean())), 'mse_gain': float((old[mask]-avg[mask]).mean())}
    keep = np.argsort(old)[:-5]
    r['gain_without_worst5_baseline'] = float(np.sqrt(old[keep].mean())-np.sqrt(avg[keep].mean()))
    if bootstrap:
        ix = np.random.default_rng(90020261012).integers(0, 400, (bootstrap, 400))
        b = np.sqrt(old[ix].mean(1)) - np.sqrt(avg[ix].mean(1))
        r['conditional_id_bootstrap95'] = np.quantile(b, [.025, .975]).tolist()
    r['uncertainty_limit'] = 'Conditions on reused data, features and fitted models. Does not include adaptive feature/model selection or refitting uncertainty.'
    return r


if __name__ == '__main__':
    print(json.dumps(evaluate(prepare_base()['baseline']), indent=2))
