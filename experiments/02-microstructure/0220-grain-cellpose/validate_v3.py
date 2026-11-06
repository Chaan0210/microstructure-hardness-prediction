"""v3 validation (plan-v3.json): arms A (v2 features on masks-v3) and B (v3 features on masks-v3) vs the v2 blend."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, sys, warnings
import numpy as np
from pathlib import Path
from PIL import Image
warnings.filterwarnings('ignore')
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(R))
from features_v3 import feature_matrix_v3
from models_v3 import W, MIN_OBJ, select_and_fit

OUT = _repo_path(R, 'validation-v3.json')
assert not OUT.exists(), 'refuse overwrite'
fp = _repo_path(R, 'features-v3-train.npz')
if not fp.exists():
    z = np.load(_repo_path(R, 'masks-v3-train.npz'))
    imgs = [np.array(Image.open(_repo_path(P, f'data/train/{i}.png'))) for i in z['ids']]
    X, keys = feature_matrix_v3(imgs, z['masks']); np.savez(_repo_io(fp), ids=z['ids'], X=X, keys=np.array(keys))
F = np.load(_repo_io(fp)); keys3 = list(F['keys']); pos = {i: k for k, i in enumerate(F['ids'])}
keys2 = list(np.load(_repo_path(R, 'features-train.npz'))['keys'])
arms = {'A_v2feat_v3mask': [keys3.index(k) for k in keys2], 'B_v3feat_v3mask': list(range(len(keys3)))}
v2 = np.load(_repo_path(R, 'oof-v2.npz'))
splits = {'original': (_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), 'fixed_control'),
          '20261031': (_repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), 'best021'),
          '20261101': (_repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz'), 'best021')}
out = {'n_features': {a: len(c) for a, c in arms.items()}}; per = {}
for arm, cols in arms.items():
    kk = [keys3[c] for c in cols]; out[arm] = {}; per[arm] = {}
    for name, (path, col) in splits.items():
        d = np.load(_repo_io(path)); ids, y, fold, p021 = d['ids'], d['y'], d['fold'], d[col]
        assert (v2[f'{name}_ids'] == ids).all()
        base = v2[f'{name}_blend']
        X = F['X'][[pos[i] for i in ids]][:, cols]
        failed = np.nan_to_num(X[:, kk.index('n_obj')], nan=0) < MIN_OBJ
        g = np.zeros(len(y)); fams = []
        for k in range(5):
            tr, va = fold != k, fold == k
            best, scores, predict, _ = select_and_fit(X[tr], y[tr], fold[tr], kk)
            fams.append(best); g[va] = predict(X[va])
        g = np.where(failed, p021, g); b = (1 - W) * p021 + W * g
        rm = lambda a: float(np.sqrt(np.mean((a - y) ** 2)))
        out[arm][name] = dict(grain_standalone=rm(g), blend=rm(b), v2_blend=rm(base), best021=rm(p021), gain_vs_v2=rm(base) - rm(b), gain_vs_021=rm(p021) - rm(b),
                              folds_better_than_v2=int(sum(np.sqrt(np.mean((b[fold == k] - y[fold == k]) ** 2)) < np.sqrt(np.mean((base[fold == k] - y[fold == k]) ** 2)) for k in range(5))),
                              families=fams, n_failed=int(failed.sum()))
        per[arm][name] = dict(ids=ids, y=y, b=b, base=base, g=g)
        print(arm, name, json.dumps({k: v for k, v in out[arm][name].items() if k != 'families'}), fams, flush=True)
    ref = per[arm]['original']['ids']
    def se(n, key):
        ix = {i: k for k, i in enumerate(per[arm][n]['ids'])}; j = [ix[i] for i in ref]; return (per[arm][n][key][j] - per[arm][n]['y'][j]) ** 2
    s0 = np.mean([se(n, 'base') for n in splits], 0); s1 = np.mean([se(n, 'b') for n in splits], 0)
    rng = np.random.default_rng(20261008); gains = []
    for _ in range(10000):
        i = rng.integers(0, len(ref), len(ref)); gains.append(np.sqrt(s0[i].mean()) - np.sqrt(s1[i].mean()))
    pooled = dict(v2_blend=float(np.sqrt(s0.mean())), blend=float(np.sqrt(s1.mean())), gain_vs_v2=float(np.sqrt(s0.mean()) - np.sqrt(s1.mean())),
                  ci95_descriptive=[float(np.quantile(gains, .025)), float(np.quantile(gains, .975))])
    out[arm]['pooled'] = pooled
    out[arm]['qualifies'] = bool(all(out[arm][n]['gain_vs_v2'] > 0 for n in splits) and pooled['gain_vs_v2'] >= 0.10)
    print(arm, 'pooled', json.dumps(pooled), 'qualifies', out[arm]['qualifies'], flush=True)
q = [a for a in arms if out[a]['qualifies']]
out['adopted'] = max(q, key=lambda a: out[a]['pooled']['gain_vs_v2']) if q else None
out['plan'] = 'plan-v3.json'
json.dump(out, open(_repo_io(OUT), 'x'), indent=1)
np.savez(_repo_path(R, 'oof-v3.npz'), **{f'{a}_{n}_{k}': v for a, dd in per.items() for n, dct in dd.items() for k, v in dct.items()})
print('ADOPTED', out['adopted'])
