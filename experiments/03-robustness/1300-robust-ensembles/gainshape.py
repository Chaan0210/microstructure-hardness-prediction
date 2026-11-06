"""Gain of noisy-row changes by label-free prediction extremeness (|061 prediction - noisy-row median|, per partition quintiles)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, nsgain as N
exec(open(_repo_io('sudev.py')).read().split("su(c53, c57")[0].split('def su(')[0])
exec('def su' + open(_repo_io('sudev.py')).read().split('def su')[1].split("c53 = np.where")[0])
exec("c53 = np.where" + open(_repo_io('sudev.py')).read().split("c53 = np.where")[1].split("su(c53, c57")[0])
def shape(nm, base, new, pub):
    out = np.zeros(5); cnt = np.zeros(5)
    for p in range(3):
        b = base[p, g]; dv = np.abs(b - np.median(b)); q = np.searchsorted(np.quantile(dv, [.2, .4, .6, .8]), dv)
        d = (new[p, g] - y[g]) ** 2 - (b - y[g]) ** 2
        for k in range(5): out[k] += d[q == k].sum(); cnt[k] += (q == k).sum()
    print(f'{nm:10s} dMSE per row by extremeness quintile (central -> extreme): ' + ' '.join(f'{v:+6.2f}' for v in out / cnt) + f'   {pub}')
shape('053->057', c53, c57, 'Public -0.042'); shape('053->061', c53, c61, 'Public -0.055'); shape('061->064', c61, c64, 'Public +0.034')
shape('061->069', c61, c69, 'Public +0.019'); shape('061->K2', c61, k2, 'n/a')
import csv
t61 = np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../artifacts/submission-candidate-061.csv')))]); T = np.load(_repo_io('../push-1200/fin-d1234n12.npz')); tg = np.nan_to_num(T['noise'], nan=0) >= 12
dv_dev = np.concatenate([np.abs(c61[p, g] - np.median(c61[p, g])) for p in range(3)]); dv_te = np.abs(t61[tg] - np.median(t61[tg]))
print('diagnostic: share of test noisy rows beyond the dev 60/80% extremeness cutoffs:', np.round([np.mean(dv_te > np.quantile(dv_dev, q)) for q in (.6, .8)], 3), '(dev: 0.4, 0.2)')
