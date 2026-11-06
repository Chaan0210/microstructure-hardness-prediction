"""Label-supervised fine-tuning of the u1 U-Net phase readout for high-noise images (>= 12 gray), gate 1 (linear, fold-honest).
Per Study900 partition/fold, using outer-training rows only:
  1. hardness function on clean+mid training rows: RidgeCV on the non-phase columns of X98 + u_ph;  beta = its u_ph coefficient
  2. implied phase of the high-noise training rows: phi* = u_ph + (y - g(x)) / beta
  3. readout: phase-vs-matrix logit d = d0(h) + D.h + c (D in R^32, c scalar; d0 = original u1 readout on the pooled penultimate maps);
     f(D, c) = sum g0 sigmoid(d) / sum g0 (g0 = original matrix+phase probability) -> equals u_ph at D = 0, c = 0.
     fit min mean (f - phi*)^2 + lam |(D, c)|^2 on the high-noise training rows; lam by inner 3-fold CV (grid incl. lam -> inf = no change)
  4. new column u_ph_ft = f for high-noise rows (u_ph elsewhere) -> RidgeCV on X98 with u_ph replaced, scored on outer-validation rows
Control: phi* permuted among the high-noise training rows. usage: ft_readout.py [shuffle]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, warnings, numpy as np, torch
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore'); torch.set_num_threads(8)
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sys.path.insert(0, str(_repo_path(E, 'push-900')))
from blend_dev import load
SHUF = len(sys.argv) > 1 and sys.argv[1] == 'shuffle'
B = load(3); ids = B['ids']; Y = B['y']; FOLD = B['fold']
S9 = np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True); X = S9['X98'].astype(float); K = [str(k) for k in S9['keys98']]
NZ = np.nan_to_num(S9['noise'], nan=0); HN = NZ >= 12
P = np.load(_repo_path(H, 'pen-u1-train.npz')); pid = [str(i) for i in P['ids']]; pos = [pid.index(i) for i in ids]
Hm = torch.tensor(P['F'][pos].astype(np.float32)).reshape(len(ids), 32, -1)                # [400, 32, 16384]
W, b = torch.tensor(P['W']), torch.tensor(P['b'])
L = torch.einsum('kc,ncp->nkp', W, Hm) + b[None, :, None]; pr = L.softmax(1)
G0 = pr[:, 0] + pr[:, 1]; D0 = L[:, 1] - L[:, 0]                                             # weights, original phase-vs-matrix logit
Hs = Hm / Hm.std(dim=(0, 2), keepdim=True)                                                   # per-channel scaled maps for the correction
UPH = (G0 * torch.sigmoid(D0)).sum(1) / G0.sum(1); kph = K.index('u_ph')
print('pooled u_ph vs X98 u_ph corr', np.corrcoef(UPH.numpy(), np.nan_to_num(X[:, kph]))[0, 1].round(4), flush=True)
PHASE = [c for c in K if any(s in c for s in ('u_ph', 'm_ph', 'z_f', 'sh_cor', 'sp_', 'spfrac', 'dark_span'))]
NONPH = [K.index(c) for c in K if c not in PHASE]
def ridge(): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 17)))
def feat(rows, Dc):
    D, c = Dc; d = D0[rows] + torch.einsum('c,ncp->np', D, Hs[rows]) + c; return (G0[rows] * torch.sigmoid(d)).sum(1) / G0[rows].sum(1)
def fit(rows, target, lam, iters=60):
    D = torch.zeros(32, requires_grad=True); c = torch.zeros(1, requires_grad=True); t = torch.tensor(target, dtype=torch.float32)
    opt = torch.optim.LBFGS([D, c], lr=0.5, max_iter=iters, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = ((feat(rows, (D, c)) - t) ** 2).mean() + lam * ((D ** 2).sum() + (c ** 2).sum()); loss.backward(); return loss
    opt.step(closure); return D.detach(), c.detach()
LAMS = [1e-5, 1e-4, 1e-3, 1e-2, None]                                                       # None = no change (D = 0)
rng = np.random.RandomState(7)
OUT = {'base': np.zeros((3, 400)), 'ft': np.zeros((3, 400))}; PH = np.zeros((3, 400)); picks = []
for k in range(3):
    for f in range(5):
        tr, va = FOLD[k] != f, FOLD[k] == f; cm = tr & ~HN; hn = np.where(tr & HN)[0]
        Zc = np.c_[X[:, NONPH], X[:, kph]]; g = ridge().fit(Zc[cm], Y[cm])
        beta = g[-1].coef_[-1] / g[1].scale_[-1]                                               # raw-unit slope of u_ph
        phi = X[hn, kph] + (Y[hn] - g.predict(Zc[hn])) / beta
        if SHUF: phi = phi[rng.permutation(len(phi))]
        sc = np.zeros(len(LAMS))
        for itr, iva in KFold(3, shuffle=True, random_state=100 * k + f).split(hn):
            for j, lam in enumerate(LAMS):
                Dc = (torch.zeros(32), torch.zeros(1)) if lam is None else fit(hn[itr], phi[itr], lam)
                sc[j] += ((feat(hn[iva], Dc).numpy() - phi[iva]) ** 2).sum()
        j = int(np.argmin(sc)); lam = LAMS[j]; picks.append(lam); zero = (torch.zeros(32), torch.zeros(1))
        fnew = feat(np.arange(400), zero if lam is None else fit(hn, phi, lam)).numpy()          # validation rows: fit on all training rows
        for itr, iva in KFold(3, shuffle=True, random_state=100 * k + f).split(hn):            # training rows: cross-fitted (no own label)
            fnew[hn[iva]] = feat(hn[iva], zero if lam is None else fit(hn[itr], phi[itr], lam)).numpy()
        ph = np.where(HN, fnew, UPH.numpy()); PH[k, va] = ph[va]
        Xb = X.copy(); Xf = X.copy(); Xf[:, kph] = np.where(HN, fnew - UPH.numpy() + X[:, kph], X[:, kph])
        OUT['base'][k, va] = ridge().fit(Xb[tr], Y[tr]).predict(Xb[va]); OUT['ft'][k, va] = ridge().fit(Xf[tr], Y[tr]).predict(Xf[va])
    print('partition', k, 'lam picks so far', picks[-5:], flush=True)
r = lambda m, msk: np.sqrt(((OUT[m][:, msk] - Y[msk]) ** 2).mean())
print(('SHUFFLED CONTROL ' if SHUF else '') + 'Ridge on X98, outer-validation RMSE  all: base %.4f ft %.4f | high-noise: base %.4f ft %.4f' % (r('base', np.ones(400, bool)), r('ft', np.ones(400, bool)), r('base', HN), r('ft', HN)))
print('per partition high-noise gain', [round(float(np.sqrt(((OUT['base'][k, HN] - Y[HN]) ** 2).mean()) - np.sqrt(((OUT['ft'][k, HN] - Y[HN]) ** 2).mean())), 3) for k in range(3)])
print('corr with y on high-noise rows: original u_ph %.3f, fine-tuned (out-of-fold, mean over partitions) %.3f' % (np.corrcoef(X[HN, kph], Y[HN])[0, 1], np.mean([np.corrcoef(PH[k, HN], Y[HN])[0, 1] for k in range(3)])))
print('lambda picks', {str(l): picks.count(l) for l in LAMS})
np.savez(_repo_path(H, f'ft-readout{"-shuf" if SHUF else ""}.npz'), base=OUT['base'], ft=OUT['ft'], ph=PH)
