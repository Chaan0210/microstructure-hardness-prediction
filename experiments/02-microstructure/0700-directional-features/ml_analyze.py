from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, itertools
from scipy.optimize import nnls
O = np.load(_repo_io('mlcmp-oof.npz')); parts = ['original', '20261031', '20261101']
D1 = np.load(_repo_io('../push-600/dev-u1dmavgH2Z-angh.npz'), allow_pickle=True)
D4 = [np.load(_repo_io(f'../push-600/dev-{v}-angh.npz'), allow_pickle=True) for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']]
names = ['ridge', 'svr', 'rf', 'et', 'lgbm', 'xgb', 'catboost']
res = {}
def pooled(getpred, W=0.75):
    se = []; seg = []
    for p in parts:
        y = D1[p + '_y']; q = D1[p + '_p021']; f = D1[p + '_failed']; g = getpred(p)
        se.append(np.mean((0.25 * q + 0.75 * np.where(f, q, g) - y) ** 2) if W == 0.75 else np.mean(((1 - W) * q + W * np.where(f, q, g) - y) ** 2)); seg.append(np.mean((g - y) ** 2))
    return np.sqrt(np.mean(seg)), np.sqrt(np.mean(se))
tab1 = lambda p: D1[p + '_g']; tab4 = lambda p: np.mean([d[p + '_g'] for d in D4], 0)
print('%-28s %8s %8s %8s' % ('model', 'grain', 'W0.75', 'errcorr(TabPFN)'))
for nm, fn in [('TabPFN-3.5 (avg var, 3 seeds)', tab1), ('TabPFN-3.5 039 (4 var)', tab4)] + [(n, (lambda n: lambda p: O[f'{p}|{n}'])(n)) for n in names]:
    gr, bl = pooled(fn); ec = np.mean([np.corrcoef(fn(p) - D1[p + '_y'], tab1(p) - D1[p + '_y'])[0, 1] for p in parts])
    print('%-28s %8.3f %8.3f %8.3f' % (nm, gr, bl, ec))
print('--- ensembles (grain level, then W0.75 with 021)')
avg_cls = lambda p: np.mean([O[f'{p}|{n}'] for n in names], 0)
avg_gbm = lambda p: np.mean([O[f'{p}|{n}'] for n in ['lgbm', 'xgb', 'catboost']], 0)
for nm, fn in [('avg of 7 classic', avg_cls), ('avg of 3 GBMs', avg_gbm), ('TabPFN4*0.8 + ridge*0.2', lambda p: 0.8 * tab4(p) + 0.2 * O[f'{p}|ridge']),
               ('TabPFN4*0.8 + catboost*0.2', lambda p: 0.8 * tab4(p) + 0.2 * O[f'{p}|catboost']), ('TabPFN4*0.7 + ridge*0.15 + cat*0.15', lambda p: 0.7 * tab4(p) + 0.15 * O[f'{p}|ridge'] + 0.15 * O[f'{p}|catboost']),
               ('TabPFN4*0.5 + avg7*0.5', lambda p: 0.5 * tab4(p) + 0.5 * avg_cls(p))]:
    print('%-36s grain %.3f  W0.75 %.3f' % ((nm,) + pooled(fn)))
# nested stacking: NNLS weights over [TabPFN4, 7 classic, 021] fit on 4 folds of OOF, applied to held-out fold (final blend directly, no extra W)
se = []; wts = []
for p in parts:
    y = D1[p + '_y']; q = D1[p + '_p021']; f = D1[p + '_failed']; fold = D1[p + '_fold']
    M = np.column_stack([np.where(f, q, tab4(p))] + [np.where(f, q, O[f'{p}|{n}']) for n in names] + [q]); out = np.zeros(len(y))
    for k in range(5):
        tr = fold != k; w, _ = nnls(M[tr], y[tr]); w = w / w.sum(); out[fold == k] = M[fold == k] @ w; wts.append(w)
    se.append(np.mean((out - y) ** 2))
print('nested NNLS stack over [TabPFN4, %s, 021]: pooled %.3f' % (','.join(names), np.sqrt(np.mean(se))))
print('mean weights:', ' '.join('%s %.2f' % (n, w) for n, w in zip(['TabPFN4'] + names + ['021'], np.mean(wts, 0))))
