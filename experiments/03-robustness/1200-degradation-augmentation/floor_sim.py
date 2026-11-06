"""Measurement-floor experiment: same 249 clean train images and labels, TabPFN-3.5 5-fold CV (2 repeats) on
(A) their clean production features, (B) features measured on the label-free degraded versions (deg, deg2, deg3),
vs (C) TabPFN CV on the 251 real noisy-half rows alone. Difference A->B = information lost to (simulated) degradation at clean label noise."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.model_selection import KFold
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; fid = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(fid)}
def cv(X, y, reps=(0, 1)):
    out = []
    for r in reps:
        o = np.zeros(len(y))
        for tr, va in KFold(5, shuffle=True, random_state=r).split(X):
            m = harness.tabpfn_new('v3.5', r, 'mps'); m.fit(X[tr], y[tr]); o[va] = m.predict(X[va])
        out.append(o)
    return np.array(out)
rm = lambda o, y: float(np.sqrt(((o - y) ** 2).mean()))
S = np.load(_repo_io('../push-500/synthrows-train_deg3-u1dmavgH2Z.npz'), allow_pickle=True); sid = [str(i) for i in S['ids']]
y = np.array([lab[i] for i in sid]); XA = F['X'][[pos[i] for i in sid]].astype(float)
oA = cv(XA, y); print('A clean features (249 clean rows):', rm(oA, y), flush=True)
for tag in ('deg', 'deg2', 'deg3'):
    S = np.load(_repo_io(f'../push-500/synthrows-train_{tag}-u1dmavgH2Z.npz'), allow_pickle=True); assert [str(i) for i in S['ids']] == sid
    oB = cv(S['X'].astype(float), y); print(f'B {tag} degraded features (same rows/labels):', rm(oB, y), flush=True)
nz = F['X'][:, keys.index('noise')]; med = np.nanmedian(nz); noisy = [i for i in fid if i not in set(sid)]
yC = np.array([lab[i] for i in noisy]); XC = F['X'][[pos[i] for i in noisy]].astype(float)
oC = cv(XC, yC); print('C real noisy-half rows alone (251):', rm(oC, yC), ' label sd', yC.std(), ' (clean-half label sd', y.std(), ')', flush=True)
np.savez(_repo_io('floor_sim.npz'), oA=oA, y=y, oC=oC, yC=yC)
