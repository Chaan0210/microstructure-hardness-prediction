"""Nonlinear stacking: TabPFN meta-learner on the 073 component OOF predictions (cnn, tab, ridge, logr, set, gmm) + noise + u_ph, predicting y.
10-fold CV within each partition (component predictions are already out-of-fold). Variants: predict y directly; predict residual y - 073.
Reports RMSE vs 073 per band. CPU."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300')); import harness
from sklearn.model_selection import KFold
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load, align_dev
import nsgain as N
y, g, C, nz, fail = N.y, N.g, N.C, N.nz, N.fail; f = fail[0]; ng = ~g & ~f
S = ['038', '039', '044']; V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']; Tn = np.mean([align_dev(f'abl/dev-nb-{s}-{v}.npz') for s in S for v in V], 0); Z = np.load(_repo_io('nb-partners.npz'))
Cx = dict(C); Cx['tab'] = np.where(ng, Tn, C['tab'])
for nm, key in (('ridge', 'ridge'), ('logr', 'logridge'), ('gmm', 'gmm')): Cx[nm] = np.where(ng, C[nm] + (Z[key + '_nb_dev'] - Z[key + '_full_dev']), C[nm])
c73 = N.blend(Cx); sid = load(3)['ids']
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); k = [str(x) for x in F['keys']]; pos = {str(i): j for j, i in enumerate(F['ids'])}
uph = F['X'][[pos[i] for i in sid], k.index('u_ph')].astype(float)
out = {m: c73.copy() for m in ('direct', 'resid')}
for p in range(3):
    Xm = np.c_[np.array([Cx[c][p] for c in N.W]).T, nz, uph]; ok = ~f
    for a, b in KFold(10, shuffle=True, random_state=p).split(np.where(ok)[0]):
        ia, ib = np.where(ok)[0][a], np.where(ok)[0][b]
        m = harness.tabpfn_new('v3.5', p, 'cpu'); m.fit(Xm[ia], y[ia]); out['direct'][p, ib] = m.predict(Xm[ib])
        m = harness.tabpfn_new('v3.5', p, 'cpu'); m.fit(Xm[ia], (y - c73[p])[ia]); out['resid'][p, ib] = c73[p, ib] + m.predict(Xm[ib])
    print('partition', p, flush=True)
rb = lambda a, msk: np.sqrt(((a - y) ** 2)[:, msk].mean())
for nm, a in (('073', c73), ('meta direct', out['direct']), ('meta resid', out['resid'])):
    print(f'{nm:12s} all {N.rm(a):.4f} clean<11 {rb(a, (nz < 11) & ~f):.3f} 11-12 {rb(a, ng & (nz >= 11)):.3f} noisy {rb(a, g):.3f} parts {np.round(np.sqrt(((a - y) ** 2).mean(1)), 3)}')
