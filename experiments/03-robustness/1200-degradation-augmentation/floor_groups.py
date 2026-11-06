"""Which feature group carries the degradation loss? Same 249 clean rows/labels; start from clean features and swap ONE group to its
deg2-degraded measurement (and the reverse: start from deg2 and restore one group to clean). TabPFN-3.5 5-fold CV x2."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, re, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.model_selection import KFold
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
tag = sys.argv[1] if len(sys.argv) > 1 else 'deg2'
S = np.load(_repo_io(f'../push-500/synthrows-train_{tag}-u1dmavgH2Z.npz'), allow_pickle=True); sid = [str(i) for i in S['ids']]
y = np.array([lab[i] for i in sid]); XA = F['X'][[pos[i] for i in sid]].astype(float); XB = S['X'].astype(float)
def grp(k):
    if k == 'noise': return 'noise'
    if re.search(r'(u_ph|m_ph|z_f|^sp_|spfrac|dark_span|f_dark|sp_frac)', k): return 'phase'
    if re.search(r'(het4|sizefield|tile_logA|tile8|m_la_std|m_la_f|m_la_tile|logA_std|eqd_q90_q10|logA_skew|logA_bimod|vstd_sizefield|vstd_tile8|vstd_grain_logA)', k): return 'het'
    if re.search(r'(m_el|ar_|orient|stcoh|solidity|boundary_coh|vstd_ar|vstd_orient|vstd_solid)', k): return 'elong'
    if re.search(r'(pore)', k): return 'pore'
    if re.search(r'(m_la_mean|grain_|n_obj|obj_per|aw_|eqd|coarse|fine|neighbors|density|mean_area|vstd_)', k): return 'size'
    return 'other'
G = np.array([grp(k) for k in keys]); print({g: int((G == g).sum()) for g in np.unique(G)})
print({g: [k for k, gg in zip(keys, G) if gg == g] for g in ('other',)})
def cv(X, reps=(0, 1)):
    out = []
    for r in reps:
        o = np.zeros(len(y))
        for tr, va in KFold(5, shuffle=True, random_state=r).split(X):
            m = harness.tabpfn_new('v3.5', r, 'mps'); m.fit(X[tr], y[tr]); o[va] = m.predict(X[va])
        out.append(o)
    return float(np.sqrt(((np.array(out) - y) ** 2).mean()))
print('clean', round(cv(XA), 3), ' all degraded', round(cv(XB), 3), flush=True)
for g in ('phase', 'size', 'het', 'elong', 'pore', 'other', 'noise'):
    m = G == g
    X1 = XA.copy(); X1[:, m] = XB[:, m]; X2 = XB.copy(); X2[:, m] = XA[:, m]
    print(f'{g:6s} degrade only this group: {cv(X1):.3f}   restore only this group: {cv(X2):.3f}', flush=True)
