"""Domain-knowledge model and post-processing checks (train-only).
(1) Physics-structured hardness model, fitted per fold by nonlinear least squares:
    H = (a + b*f + c*s + d*f*s) * exp(-n*p) * (1 + e*el)
    f = U-Net phase fraction (u_ph), s = area-weighted 1/sqrt(d) (Hall-Petch, aw_inv_sqrt_d), p = pixel pore fraction (px_pore_frac), el = elongation (m_el_pix).
    Variants: plain (all real rows), DTA (+ deg..deg4 copies, used on noise >= 12 rows). Evaluated alone, as a cross-fitted extra partner on 073.
(2) Residual structure of 073 by predicted-value decile and by physics-variable extremes (clean / noisy), with bootstrap SE."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.optimize import least_squares
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
import nsgain as N
P = Path(_repo_io('../..')); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); k = [str(x) for x in F['keys']]; pos = {str(i): j for j, i in enumerate(F['ids'])}
cols = ['u_ph', 'aw_inv_sqrt_d', 'px_pore_frac', 'm_el_pix']; ci = [k.index(c) for c in cols]; ni = k.index('noise')
def Z(X): Z_ = X[:, ci].astype(float); return np.where(np.isfinite(Z_), Z_, np.nanmedian(Z_, 0))
XA = Z(F['X'])
SX, SID = [], []
for s in ['deg', 'deg2', 'deg3', 'deg4']:
    S = np.load(_repo_io(f'../push-1200/synthrows-{s}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(Z(S['X'][m])); SID += [str(i) for i in np.array(S['ids'])[m]]
SX = np.vstack(SX); SID = np.array(SID)
def model(th, X): f, s, p, el = X.T; a, b, c, d, n, e = th; return (a + b * f + c * s + d * f * s) * np.exp(-n * p) * (1 + e * el)
def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    r = least_squares(lambda th: model(th, Xs) - y, x0=[y.mean(), 5, -5, 0, 0, 0], loss='soft_l1', f_scale=15); return lambda Xn: model(r.x, (Xn - mu) / sd)
O = {'plain': np.zeros((3, 400)), 'dta': np.zeros((3, 400))}
for p_, path in enumerate(splits):
    d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
    for kk in range(5):
        tr = [i for i, f in zip(ids, fold) if f != kk]; va = [i for i, f in zip(ids, fold) if f == kk]; vs = set(va); jv = [pos[i] for i in va]
        Xt, yt = XA[[pos[i] for i in tr]], np.array([lab[i] for i in tr]); O['plain'][p_, [sid.index(i) for i in va]] = fit(Xt, yt)(XA[jv])
        kp = np.array([s not in vs for s in SID]); O['dta'][p_, [sid.index(i) for i in va]] = fit(np.vstack([Xt, SX[kp]]), np.r_[yt, [lab[s] for s in SID[kp]]])(XA[jv])
y, g, nz, fail, fold, C = N.y, N.g, N.nz, N.fail, N.fold, N.C; f = fail[0]; c73 = np.load(_repo_io('dev-nb-candidate.npy'))
phys = np.where(g, O['dta'], O['plain']); phys = np.where(fail, c73, phys)
rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
print('physics model alone: clean %.3f noisy %.3f all %.3f | 073 clean %.3f noisy %.3f | Ridge(98) clean %.3f' % (rb(phys, (nz < 11) & ~f), rb(phys, g), rb(phys, ~f), rb(c73, (nz < 11) & ~f), rb(c73, g), rb(C['ridge'], (nz < 11) & ~f)))
print('error corr physics vs 073 %.3f' % np.corrcoef((phys - y)[:, ~f].ravel(), (c73 - y)[:, ~f].ravel())[0, 1])
grid = np.linspace(0, 0.5, 51); cf = c73.copy(); ch = []
for p in range(3):
    for kk in range(5):
        tr = (fold[p] != kk) & ~f; te = fold[p] == kk; dd = phys[p] - c73[p]; r = c73[p] - y
        a = grid[np.argmin([((r[tr] + aa * dd[tr]) ** 2).mean() for aa in grid])]; ch.append(a); cf[p, te] = c73[p, te] + a * dd[te]
pp = np.sqrt(((c73 - y) ** 2).mean(1)) - np.sqrt(((cf - y) ** 2).mean(1)); print(f'cross-fitted extra-partner share {np.mean(ch):.3f} (sd {np.std(ch):.3f}) -> gain {N.rm(c73) - N.rm(cf):+.4f} parts {np.round(pp, 3)}')
# (2) residual structure of 073 at prediction / physics extremes
r = (y - c73).mean(0); pm = c73.mean(0); rng = np.random.RandomState(0)
def cell(m): v = r[m]; bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(2000)]; return f'{v.mean():+.2f}±{np.std(bs):.2f} (n {m.sum()})'
for nm, band in (('clean<11', (nz < 11) & ~f), ('noisy>=12', g)):
    q = np.quantile(pm[band], [0.1, 0.9]); print(f'{nm}: mean residual y-073 at predicted bottom 10% {cell(band & (pm <= q[0]))}, top 10% {cell(band & (pm >= q[1]))}')
    for j, c in enumerate(cols):
        x = XA[[pos[i] for i in sid], j]; q = np.quantile(x[band], [0.1, 0.9]); print(f'    {c:14s} low 10% {cell(band & (x <= q[0]))}  high 10% {cell(band & (x >= q[1]))}')
