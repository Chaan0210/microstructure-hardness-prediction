"""Does adding label-carrying degraded copies of clean TRAIN images (push-500 synthrows, degradation of the image only, label = the original
image's label) to the TabPFN context improve REAL noisy validation rows? Official dev partitions/folds (as dev900.py), 038-slot features (H2Z).
Sim rows whose SOURCE image is in the validation fold are excluded from that fold's context (no label of a validation image is used).
usage: synaug_dev.py VARIANT OPTION DEVICE [SEEDS]   OPTION: base | deg2 | deg+deg2 | deg2flag | deg2+deg3
-> synaug-<VARIANT>-<OPTION>.npz with OOF g for all rows ([3, 400] in each partition's id order)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
P = Path(_repo_io('../..')); v, opt, device = sys.argv[1], sys.argv[2], sys.argv[3]; seeds = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else '0,1,2').split(',')]
F = np.load(_repo_io(f'../push-500/feat-{v}-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
lab = {r.split(',')[0]: float(r.split(',')[1]) for r in open(_repo_path(P, 'data/train.csv')).read().split()[1:]}
TRMIN = float(__import__('os').environ.get('TRMIN', '-1'))
tags = {'base': [], 'deg2': ['deg2'], 'deg+deg2': ['deg', 'deg2'], 'deg2flag': ['deg2'], 'deg2+deg3': ['deg2', 'deg3'], 'all3': ['deg', 'deg2', 'deg3']}[opt]; flag = opt.endswith('flag')
SX, SY, SID = [], [], []
for t in tags:
    S = np.load(_repo_io(f'../push-500/synthrows-train_{t}-{v}.npz'), allow_pickle=True); assert [str(k) for k in S['keys']] == keys
    SX.append(S['X'].astype(float)); SID += [str(i) for i in S['ids']]; SY += [lab[str(i)] for i in S['ids']]
SX = np.vstack(SX) if SX else np.zeros((0, len(keys))); SY = np.array(SY); SID = np.array(SID)
NMIN = float(__import__('os').environ.get('NOISEMIN', '-1'))
if NMIN > 0 and len(SX):
    _k = [str(x) for x in np.load(_repo_io(f'../push-500/synthrows-train_{tags[0]}-{v}.npz'), allow_pickle=True)['keys']]; _m = SX[:, _k.index('noise')] * 255 >= NMIN; SX, SY, SID = SX[_m], SY[_m], SID[_m]
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'),
          '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
save = {}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; g = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k; tr = tr & (np.nan_to_num(X[:, keys.index('noise')] * 255, nan=0) >= TRMIN); vids = set(np.array(ids)[va])
        keep = np.array([s not in vids for s in SID], bool) if len(SID) else np.zeros(0, bool)
        Xt = np.vstack([X[tr], SX[keep]]); yt = np.r_[y[tr], SY[keep]]; Xv = X[va]
        if flag: Xt = np.c_[Xt, np.r_[np.zeros(tr.sum()), np.ones(keep.sum())]]; Xv = np.c_[Xv, np.zeros(va.sum())]
        ps = []
        for s in seeds:
            m = harness.tabpfn_new('v3.5', 100 * s + k, device); m.fit(Xt, yt); ps.append(m.predict(Xv))
        g[va] = np.mean(ps, 0)
    nz = X[:, keys.index('noise')] * 255; noisy = nz >= 12
    print(name, opt, 'all %.4f noisy %.4f clean %.4f' % (np.sqrt(((g - y) ** 2).mean()), np.sqrt(((g - y)[noisy] ** 2).mean()), np.sqrt(((g - y)[nz < 9.5] ** 2).mean())), flush=True)
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_g': g})
np.savez(_repo_io(f'synaug-{v}-{opt}' + (f'-tr{TRMIN:g}' if TRMIN > 0 else '') + (f'-n{NMIN:g}' if NMIN > 0 else '') + '.npz'), **save)
