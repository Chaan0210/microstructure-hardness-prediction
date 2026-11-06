"""External-review proposal 2: restricted metric learning (one AGOP/RFM step) for a kernel partner in the Ridge slot of 053 (0.80*0.16 = 12.8%).
Gaussian-kernel ridge on Study900 X98 (median-imputed, standardised in-fold), k = exp(-d_M^2 / (c * median d_M^2)), target centred.
  fixed     M = I
  diag      M = 0.5 I + 0.5 p diag(G)/tr(G)
  lowrank2  M = 0.5 I + 0.5 p U2 L2 U2'/sum(L2)        (G = AGOP of the fixed-metric fit on the outer-training rows, one update, a-priori shrink 0.5)
(c, alpha) by inner 3-fold CV on the outer-training rows for every version. usage: rfm_dev.py  -> rfm-<version>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, warnings, numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sys.path.insert(0, str(_repo_path(E, 'push-900')))
from blend_dev import load, rmse, parts
B = load(3); B9 = load(9); Y = B['y']; FOLD = B['fold']; p021 = B['p021']; fail = B['fail']
S9 = np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True); X = S9['X98'].astype(float); NZ = np.nan_to_num(S9['noise'], nan=0)
bands = {'clean': NZ < 9.5, 'mid': (NZ >= 9.5) & (NZ < 12), 'noisy': NZ >= 12}
CS, AL = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], [1e-3, 1e-2, 1e-1, 1.0]
def prep(Xtr, Xte):
    im = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xtr); sc = StandardScaler().fit(im.transform(Xtr))
    f = lambda Z: np.nan_to_num(sc.transform(im.transform(Z))); return f(Xtr), f(Xte)
def d2(A, Bm, M): AM = A @ M; return np.maximum((AM * A).sum(1)[:, None] + ((Bm @ M) * Bm).sum(1)[None, :] - 2 * AM @ Bm.T, 0)
def fit(Z, y, M, c, a):
    D = d2(Z, Z, M); m = np.median(D[np.triu_indices(len(Z), 1)]); K = np.exp(-D / (c * m)); mu = y.mean()
    coef = np.linalg.solve(K + a * np.eye(len(Z)), y - mu); return dict(Z=Z, M=M, s=c * m, coef=coef, mu=mu)
def pred(md, Zq): return md['mu'] + np.exp(-d2(Zq, md['Z'], md['M']) / md['s']) @ md['coef']
def tune(Z, y, M, seed):
    sc = np.zeros((len(CS), len(AL)))
    for itr, iva in KFold(3, shuffle=True, random_state=seed).split(Z):
        for i, c in enumerate(CS):
            for j, a in enumerate(AL): sc[i, j] += np.sum((pred(fit(Z[itr], y[itr], M, c, a), Z[iva]) - y[iva]) ** 2)
    i, j = np.unravel_index(np.argmin(sc), sc.shape); return CS[i], AL[j]
def agop(md):
    Z = md['Z']; K = np.exp(-d2(Z, Z, md['M']) / md['s']); w = K * md['coef'][None, :]               # w[j, i] = a_i k(x_j, x_i)
    grads = (-2 / md['s']) * (w.sum(1)[:, None] * Z - w @ Z) @ md['M']                             # grad f at each training point
    return grads.T @ grads / len(Z)
OUT = {k: np.zeros((3, 400)) for k in ('fixed', 'diag', 'lowrank2')}; RID = np.zeros((3, 400)); P = X.shape[1]; picks = []
for k in range(3):
    for f in range(5):
        tr, va = FOLD[k] != f, FOLD[k] == f; Ztr, Zva = prep(X[tr], X[va]); ytr = Y[tr]; I = np.eye(P); seed = 100 * k + f
        RID[k, va] = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Ztr, ytr).predict(Zva)
        c, a = tune(Ztr, ytr, I, seed); m0 = fit(Ztr, ytr, I, c, a); OUT['fixed'][k, va] = pred(m0, Zva); G = agop(m0)
        Md = 0.5 * I + 0.5 * P * np.diag(np.diag(G)) / np.trace(G)
        ev, U = np.linalg.eigh(G); U2, L2 = U[:, -2:], ev[-2:]; Ml = 0.5 * I + 0.5 * P * (U2 * L2) @ U2.T / L2.sum()
        for nm, M in (('diag', Md), ('lowrank2', Ml)):
            c2, a2 = tune(Ztr, ytr, M, seed); OUT[nm][k, va] = pred(fit(Ztr, ytr, M, c2, a2), Zva); picks.append((nm, c2, a2))
        picks.append(('fixed', c, a)); print(k, f, 'top-2 AGOP share %.2f' % (L2.sum() / ev.sum()), flush=True)
W = lambda a_: np.where(fail, p021, a_)
def c053(r): return W(0.80 * (0.2 * p021 + 0.48 * B['g'] + 0.16 * r + 0.16 * B['logridge']) + 0.05 * B['sets'][:3].mean(0) + 0.15 * B['gmm'])
ref = c053(B['ridge']); print('own RidgeCV vs Study900 ridge: alone %.4f vs %.4f, corr %.4f' % (rmse(RID, Y), rmse(B['ridge'], Y), np.corrcoef(RID.ravel(), B['ridge'].ravel())[0, 1]))
print('053 reference', round(rmse(ref, Y), 4))
for nm, m in [('own RidgeCV', RID)] + list(OUT.items()):
    np.savez(_repo_path(H, f'rfm-{nm.replace(" ", "_")}.npz'), oof=m)
    c = c053(m); br = ' '.join(f'{n} {np.sqrt(((m[:, b] - Y[b]) ** 2).mean()):.2f}' for n, b in bands.items())
    ec = np.corrcoef((m - Y).ravel(), (B['g'] - Y).ravel())[0, 1]
    print(f'{nm:12s} alone {rmse(m, Y):.4f} ({br}; err corr with TabPFN {ec:.3f}) | Ridge slot swapped: 053 {rmse(c, Y):.4f} ({rmse(ref, Y) - rmse(c, Y):+.4f}, parts '
          + '/'.join(f'{v:+.4f}' for v in parts(ref, Y) - parts(c, Y)) + ')', flush=True)
from collections import Counter; print('picked (c, alpha):', Counter(picks).most_common(8))
