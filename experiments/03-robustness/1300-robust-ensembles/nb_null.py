"""Snooping control for the nb rule on the linear partners: dev blend gain (non-gated rows, RidgeCV + logRidge on u1dmavgH2Z; logRidge delta applied
as a ratio to the 4-variant production partner) when dropping the het+intensity columns vs dropping 13 random columns (200 draws, noise/n_obj kept)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
import nsgain as N
P = Path(_repo_io('../..')); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
FD = [(np.load(_repo_io(p)), ) for p in splits]
idx = []
for (d,) in FD:
    ids, fold = [str(i) for i in d['ids']], d['fold']; idx.append([(np.array([pos[i] for i, f in zip(ids, fold) if f != k]), np.array([pos[i] for i, f in zip(ids, fold) if f == k]), [sid.index(i) for i, f in zip(ids, fold) if f == k]) for k in range(5)])
Y = np.array([lab[str(i)] for i in F['ids']])
def oof(cols, kind):
    O = np.zeros((3, 400))
    for p_ in range(3):
        for tr, va, vs in idx[p_]:
            m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
            m.fit(XA[tr][:, cols], np.log(Y[tr]) if kind == 'log' else Y[tr]); pr = m.predict(XA[va][:, cols]); O[p_, vs] = np.exp(pr) if kind == 'log' else pr
    return O
y, g, c61, C, fail = N.y, N.g, N.c61, N.C, N.fail; ng = ~g & ~fail[0]
allc = list(range(len(keys))); R0, L0 = oof(allc, 'ridge'), oof(allc, 'log')
def gain(cols):
    R, L = oof(cols, 'ridge'), oof(cols, 'log'); Cx = dict(C); Cx['ridge'] = np.where(ng, C['ridge'] + R - R0, C['ridge']); Cx['logr'] = np.where(ng, C['logr'] * L / L0, C['logr'])
    new = N.blend(Cx); return ((new - y) ** 2 - (c61 - y) ** 2)[:, ng].mean()
G = json.load(open(_repo_io('abl/groups.json'))); drop = set(G['intensity'].split() + G['het'].split())
obs = gain([j for j, k in enumerate(keys) if k not in drop]); print('het+intensity drop: non-gated dMSE %+.3f' % obs, flush=True)
cand = [j for j, k in enumerate(keys) if k not in ('noise', 'n_obj')]; rng = np.random.RandomState(0); null = []
for t in range(200):
    dr = set(rng.choice(cand, 13, replace=False)); null.append(gain([j for j in allc if j not in dr]))
null = np.array(null); print('random 13-column drops: mean %+.3f, sd %.3f, 5%% %+.3f, min %+.3f; P(null <= obs) = %.3f' % (null.mean(), null.std(), np.quantile(null, 0.05), null.min(), (null <= obs).mean()))
np.save(_repo_io('nb-null.npy'), null)
