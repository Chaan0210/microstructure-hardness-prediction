"""Dev evaluation of the aligned (qm) structure against 054 (Study900 ids, three partitions, ID bootstrap of the pooled MSE difference).
Slots: 038 (push-300 v35 / push-900 qm), 039 (push-600 angh / anghqm), 044 (push-800 anghcor / anghcorqm); partners: ridge/logridge (ridge-qm.npz),
gmm (gmm-qm-dev.npz); the grain-set model is kept as in 054. usage: eval_qm.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from blend_dev import load, rmse, parts, S9
B = load(9); ids, y, fail, p = B['ids'], B['y'], B['fail'], B['p021']
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; PN = ['original', '20261031', '20261101']
def slot(paths):
    D = [np.load(_repo_io(f), allow_pickle=True) for f in paths]; out = np.zeros((3, 400))
    for k, n in enumerate(PN):
        ii = [str(i) for i in D[0][f'{n}_ids']]; pos = {i: j for j, i in enumerate(ii)}; out[k] = np.mean([d[f'{n}_g'] for d in D], 0)[[pos[i] for i in ids]]
    return out
O = {'38': slot([f'../push-300/dev-{v}-v35.npz' for v in V]), '39': slot([f'../push-600/dev-{v}-angh.npz' for v in V]), '44': slot([f'../push-800/dev-{v}-anghcor.npz' for v in V])}
Q = {'38': slot([f'dev-{v}-qm.npz' for v in V]), '39': slot([f'dev-{v}-anghqm.npz' for v in V]), '44': slot([f'dev-{v}-anghcorqm.npz' for v in V])}
assert np.allclose(np.mean(list(O.values()), 0)[~fail], B['g'][~fail]), 'original grain block mismatch'
R = np.load(_repo_io('ridge-qm.npz')); G = np.load(_repo_io('gmm-qm-dev.npz'))['oof']
def c54(g, r, l, gm, s=None, w=(0.77, 0.08, 0.15)):
    c51 = 0.2 * p + 0.48 * g + 0.16 * r + 0.16 * l; return np.where(fail, p, w[0] * c51 + w[1] * (B['set'] if s is None else s) + w[2] * gm)
base = c54(B['g'], B['ridge'], B['logridge'], B['gmm'])
gq = np.mean(list(Q.values()), 0); gh = 0.5 * (B['g'] + gq)
rng = np.random.default_rng(20261013); BS = rng.integers(0, 400, (5000, 400))
def report(nm, c):
    d = ((base - y) ** 2 - (c - y) ** 2).mean(0); bs = d[BS].mean(1); lo, hi = np.quantile(bs, [.025, .975])
    print(f'{nm:34s} {rmse(c, y):.4f} ({rmse(base, y) - rmse(c, y):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in parts(base, y) - parts(c, y)) + f')  dMSE {d.mean():+.3f} CI [{lo:+.3f}, {hi:+.3f}]')
for s in ('38', '39', '44'): print(f'slot {s}: orig {rmse(np.where(fail, p, O[s]), y):.4f} qm {rmse(np.where(fail, p, Q[s]), y):.4f}  parts ' + '/'.join(f'{v:+.3f}' for v in parts(np.where(fail, p, O[s]), y) - parts(np.where(fail, p, Q[s]), y)))
print(f'grain block: orig {rmse(np.where(fail, p, B["g"]), y):.4f} qm {rmse(np.where(fail, p, gq), y):.4f}')
report('054', base)
report('partners qm (ridge, logridge, gmm)', c54(B['g'], R['ridge'], R['logridge'], G))
report('grain qm only', c54(gq, B['ridge'], B['logridge'], B['gmm']))
report('all qm', c54(gq, R['ridge'], R['logridge'], G))
report('all qm, grain half orig/half qm', c54(gh, R['ridge'], R['logridge'], G))
