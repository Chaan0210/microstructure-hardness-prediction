"""External-review proposal 1: fit the log-Ridge partner for the evaluation metric (hardness RMSE), only the log-Ridge slot of 053 swapped.
Three versions on the same Study900 folds, 4 U-Net variant matrices (geometric mean as in push-900 ridge_a2_logy_var4):
  A  control   Ridge(alpha 2) on log y, prediction exp(.)
  B  smearing  A times Duan's factor mean(exp(e)) from the training rows of the fold (per variant)
  C  orig-MSE  min_b,beta mean((y - exp(b + X beta))^2) + lam*(|beta - beta_log|^2 + (b - b_log)^2), started at the log fit; lam by inner 3-fold CV
               on the outer-training rows (lam -> inf reproduces A)
usage: logridge_target.py   -> lrt-{A,B,C}.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, warnings, numpy as np
from pathlib import Path
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sys.path.insert(0, str(_repo_path(E, 'push-900')))
from blend_dev import load, rmse, parts
B = load(3); B9 = load(9); ids = B['ids']; Y = B['y']; FOLD = B['fold']; p = B['p021']; fail = B['fail']
S9 = np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True); NZ = np.nan_to_num(S9['noise'], nan=0)
bands = {'clean': NZ < 9.5, 'mid': (NZ >= 9.5) & (NZ < 12), 'noisy': NZ >= 12}
VSETS = [(lambda F: F['X'][[[str(i) for i in F['ids']].index(j) for j in ids]].astype(float))(np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True))
         for v in ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z')]
LAMS = [1e1, 1e2, 1e3, 1e4, 1e5]
def prep(Xtr, Xte):
    im = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xtr); sc = StandardScaler().fit(im.transform(Xtr))
    f = lambda Z: np.nan_to_num(sc.transform(im.transform(Z))); return f(Xtr), f(Xte)
def logfit(Z, y):
    m = Ridge(alpha=2.0).fit(Z, np.log(y)); return m.intercept_, m.coef_
def origfit(Z, y, b0, w0, lam):
    n = len(y)
    def fg(t):
        b, w = t[0], t[1:]; mu = np.exp(b + Z @ w); r = mu - y
        f = np.mean(r ** 2) + lam * (np.sum((w - w0) ** 2) + (b - b0) ** 2)
        g0 = 2 * np.mean(r * mu); gw = 2 * Z.T @ (r * mu) / n
        return f, np.r_[g0 + 2 * lam * (b - b0), gw + 2 * lam * (w - w0)]
    t = minimize(fg, np.r_[b0, w0], jac=True, method='L-BFGS-B', options={'maxiter': 500}).x; return t[0], t[1:]
OUT = {k: np.zeros((3, 400)) for k in 'ABC'}; chosen = []
for k in range(3):
    for f in range(5):
        tr, va = FOLD[k] != f, FOLD[k] == f; la, lb, lc = [], [], []
        for X in VSETS:
            Ztr, Zva = prep(X[tr], X[va]); ytr = Y[tr]; b0, w0 = logfit(Ztr, ytr)
            la.append(b0 + Zva @ w0)
            e = np.log(ytr) - (b0 + Ztr @ w0); lb.append(b0 + Zva @ w0 + np.log(np.mean(np.exp(e))))
            sc = np.zeros(len(LAMS))                                  # inner CV for lam (training rows only)
            for itr, iva in KFold(3, shuffle=True, random_state=100 * k + f).split(Ztr):
                ib, iw = logfit(Ztr[itr], ytr[itr])
                for j, lam in enumerate(LAMS):
                    cb, cw = origfit(Ztr[itr], ytr[itr], ib, iw, lam); sc[j] += np.sum((np.exp(cb + Ztr[iva] @ cw) - ytr[iva]) ** 2)
            lam = LAMS[int(np.argmin(sc))]; chosen.append(lam); cb, cw = origfit(Ztr, ytr, b0, w0, lam); lc.append(cb + Zva @ cw)
        for key, L in zip('ABC', (la, lb, lc)): OUT[key][k, va] = np.exp(np.mean(L, 0))
print('lambda chosen (counts):', {l: chosen.count(l) for l in LAMS})
ref_lr = np.load(_repo_path(E, 'push-900/diverse-ridge_a2_logy_var4.npz'))['oof']; print('A reproduces stored log-Ridge OOF: max |diff|', np.abs(OUT['A'] - ref_lr).max().round(6))
W = lambda a: np.where(fail, p, a)
def c053(lr): return W(0.80 * (0.2 * p + 0.48 * B['g'] + 0.16 * B['ridge'] + 0.16 * lr) + 0.05 * B['sets'][:3].mean(0) + 0.15 * B['gmm'])
def c054(lr): return W(0.77 * (0.2 * p + 0.48 * B['g'] + 0.16 * B['ridge'] + 0.16 * lr) + 0.08 * B9['set'] + 0.15 * B9['gmm'])
r53, r54 = c053(OUT['A']), c054(OUT['A'])
for key in 'ABC':
    m = OUT[key]; np.savez(_repo_path(H, f'lrt-{key}.npz'), oof=m, ids=np.array(ids))
    bias = ' '.join(f'{n} bias {np.mean(m[:, b] - Y[b]):+.2f}' for n, b in bands.items())
    br = ' '.join(f'{n} {np.sqrt(((m[:, b] - Y[b]) ** 2).mean()):.2f}' for n, b in bands.items())
    c, d = c053(m), c054(m)
    print(f'{key}: alone {rmse(m, Y):.4f} ({br}; {bias}) | 053-structure {rmse(c, Y):.4f} ({rmse(r53, Y) - rmse(c, Y):+.4f}, parts '
          + '/'.join(f'{v:+.4f}' for v in parts(r53, Y) - parts(c, Y)) + f') | 054-structure {rmse(d, Y):.4f} ({rmse(r54, Y) - rmse(d, Y):+.4f})', flush=True)
