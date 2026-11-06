"""Linear partner on cached DINOv2-L (504 px, 3 layers, 9216-d) image features: RidgeCV (GCV over a wide grid) and log-target variant, Study900 dev folds.
Evaluated as an extra partner on top of 073 (cross-fitted share) and by band. -> dinol-ridge.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
P = Path(_repo_io('../..')); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
F = np.load(_repo_io('../candidate021-178/train-features.npz'), allow_pickle=True); pos = {str(i): j for j, i in enumerate(F['ids'])}; XA = F['X']
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
out = {}
for nm, log in (('lin', False), ('log', True)):
    O = np.zeros((3, 400)); al = []
    for p_, path in enumerate(splits):
        d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; yt = np.array([lab[i] for i in tr])
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 6, 21))).fit(XA[[pos[i] for i in tr]], np.log(yt) if log else yt); al.append(m.steps[-1][1].alpha_)
            pr = m.predict(XA[[pos[i] for i in va]]); O[p_, [sid.index(i) for i in va]] = np.exp(pr) if log else pr
    out[nm] = O; print(nm, 'alpha median', np.median(al), flush=True)
np.savez(_repo_io('dinol-ridge.npz'), **out)
import nsgain as N
y, C, nz, fail, fold = N.y, N.C, N.nz, N.fail, N.fold; f = fail[0]; c73 = np.load(_repo_io('dev-nb-candidate.npy'))
bands = {'clean<11': (nz < 11) & ~f, 'noisy>=12': (nz >= 12) & ~f}
for nm, O in out.items():
    print(nm, ' '.join(f'{k} {np.sqrt(((O - y) ** 2)[:, m].mean()):.3f}' for k, m in bands.items()), '| err corr with 073 %.3f, with CNN021 %.3f' % (np.corrcoef((O - y)[:, ~f].ravel(), (c73 - y)[:, ~f].ravel())[0, 1], np.corrcoef((O - y)[:, ~f].ravel(), (C['cnn'] - y)[:, ~f].ravel())[0, 1]))
    grid = np.linspace(0, 0.5, 51); cf = c73.copy(); ch = []
    for p in range(3):
        for k in range(5):
            tr = (fold[p] != k) & ~f; te = fold[p] == k; d = O[p] - c73[p]; r = c73[p] - y
            a = grid[np.argmin([((r[tr] + aa * d[tr]) ** 2).mean() for aa in grid])]; ch.append(a); cf[p, te] = np.where(f[te], c73[p, te], c73[p, te] + a * d[te])
    pp = np.sqrt(((c73 - y) ** 2).mean(1)) - np.sqrt(((cf - y) ** 2).mean(1)); print(f'   cross-fitted extra-partner share {np.mean(ch):.3f} (sd {np.std(ch):.3f}) -> gain {N.rm(c73) - N.rm(cf):+.4f} parts {np.round(pp, 3)}')
