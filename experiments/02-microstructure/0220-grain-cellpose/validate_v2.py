"""Corrected validation (plan-v2.json). Nothing is fitted on best021 predictions; blend weight is fixed at 0.5."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, sys, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(R))
from models_v2 import W, MIN_OBJ, select_and_fit

OUT = _repo_path(R, 'validation-v2.json')
assert not OUT.exists(), 'refuse overwrite'
F = np.load(_repo_path(R, 'features-train.npz')); keys = list(F['keys']); pos = {i: k for k, i in enumerate(F['ids'])}
splits = {'original': (_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), 'fixed_control'),
          '20261031': (_repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), 'best021'),
          '20261101': (_repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz'), 'best021')}
out, per = {}, {}
for name, (path, col) in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold, p021 = d['ids'], d['y'], d['fold'], d[col]
    X = F['X'][[pos[i] for i in ids]]
    failed = np.nan_to_num(X[:, keys.index('n_obj')], nan=0) < MIN_OBJ
    g = np.zeros(len(y)); chosen = []
    for k in range(5):
        tr, va = fold != k, fold == k
        best, scores, predict, _ = select_and_fit(X[tr], y[tr], fold[tr])   # inner CV over the remaining 4 folds
        chosen.append(dict(fold=k, family=best, inner_rmse=scores))
        g[va] = predict(X[va])
    g = np.where(failed, p021, g)
    rm = lambda a, m=slice(None): float(np.sqrt(np.mean((a[m] - y[m]) ** 2)))
    sens = {f'w={w}': rm((1 - w) * p021 + w * g) for w in (0.3, 0.4, 0.5, 0.6)}
    b = (1 - W) * p021 + W * g
    fold_gain = [rm(p021, fold == k) - rm(b, fold == k) for k in range(5)]
    out[name] = dict(best021=rm(p021), grain_standalone=rm(g), blend=rm(b), gain=rm(p021) - rm(b), fold_gains=fold_gain,
                     folds_improved=int(sum(x > 0 for x in fold_gain)), n_failed=int(failed.sum()), families=chosen,
                     err_corr_grain_vs_best021=float(np.corrcoef(y - g, y - p021)[0, 1]), sensitivity_descriptive=sens)
    per[name] = dict(ids=ids, y=y, p021=p021, blend=b, grain=g)
    print(name, json.dumps({k: v for k, v in out[name].items() if k not in ('families',)}), [c['family'] for c in chosen], flush=True)

ref = per['original']['ids']
def se(n, key):
    ix = {i: k for k, i in enumerate(per[n]['ids'])}; j = [ix[i] for i in ref]
    return (per[n][key][j] - per[n]['y'][j]) ** 2
se0 = np.mean([se(n, 'p021') for n in per], 0); se1 = np.mean([se(n, 'blend') for n in per], 0)
rng = np.random.default_rng(20261007); gains = []
for _ in range(10000):
    i = rng.integers(0, len(ref), len(ref)); gains.append(np.sqrt(se0[i].mean()) - np.sqrt(se1[i].mean()))
d_se = se0 - se1; keep = np.ones(len(ref), bool); keep[np.argsort(d_se)[-5:]] = False
pooled = dict(best021=float(np.sqrt(se0.mean())), blend=float(np.sqrt(se1.mean())), gain=float(np.sqrt(se0.mean()) - np.sqrt(se1.mean())),
              ci95=[float(np.quantile(gains, .025)), float(np.quantile(gains, .975))],
              gain_top5_removed=float(np.sqrt(se0[keep].mean()) - np.sqrt(se1[keep].mean())),
              ids_improved=int((d_se > 0).sum()), ids_worsened=int((d_se < 0).sum()),
              note='Three partitions of the same 400 images; conditional CI, not independent validation.')
gates = {'all3_gain_positive': all(out[n]['gain'] > 0 for n in splits),
         'each_split_3of5_folds': all(out[n]['folds_improved'] >= 3 for n in splits),
         'pooled_CI_lower_positive': pooled['ci95'][0] > 0,
         'pooled_top5_removed_positive': pooled['gain_top5_removed'] > 0}
out['pooled'] = pooled; out['gates'] = gates; out['passed'] = all(gates.values()); out['plan'] = 'plan-v2.json'
json.dump(out, open(_repo_io(OUT), 'x'), indent=1)
np.savez(_repo_path(R, 'oof-v2.npz'), **{f'{n}_{k}': v for n, dct in per.items() for k, v in dct.items()})
print('pooled', json.dumps(pooled)); print('gates', gates, 'PASSED' if out['passed'] else 'FAILED')
