"""Upside of partial measurement repair (deg2 sims, same 249 rows/labels): group features = clean + f*(deg2 - clean), f = remaining error fraction."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, re, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300')); sys.path.insert(0, '.')
import harness
from sklearn.model_selection import KFold
from floor_groups_def import grp
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
S = np.load(_repo_io('../push-500/synthrows-train_deg2-u1dmavgH2Z.npz'), allow_pickle=True); sid = [str(i) for i in S['ids']]
y = np.array([lab[i] for i in sid]); XA = F['X'][[pos[i] for i in sid]].astype(float); XB = S['X'].astype(float); G = np.array([grp(k) for k in keys])
def cv(X, reps=(0, 1)):
    out = []
    for r in reps:
        o = np.zeros(len(y))
        for tr, va in KFold(5, shuffle=True, random_state=r).split(X):
            m = harness.tabpfn_new('v3.5', r, 'mps'); m.fit(X[tr], y[tr]); o[va] = m.predict(X[va])
        out.append(o)
    return float(np.sqrt(((np.array(out) - y) ** 2).mean()))
for g in sys.argv[1].split(','):
    for f in (0.5, 0.25):
        m = G == g; X = XB.copy(); X[:, m] = XA[:, m] + f * (XB[:, m] - XA[:, m]); print(f'{g} remaining error {f}: {cv(X):.3f}', flush=True)
X = XB.copy(); m = (G == 'het') | (G == 'phase'); X[:, m] = XA[:, m] + 0.5 * (XB[:, m] - XA[:, m]); print(f'het+phase 0.5: {cv(X):.3f}', flush=True)
