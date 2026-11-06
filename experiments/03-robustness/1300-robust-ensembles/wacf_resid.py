"""Does the within-matrix ACF carry information beyond 061? Pre-specified 9 features (log C(r)/C(4), r = 6,8,10,13,16,20,25; log C(4); log(C(0)-C(4)))
-> RidgeCV predicting 061 dev OOF residuals, repeated 10-fold CV within each band (train-only). Reports CV R^2 and the blend gain if the predicted
residual correction is applied (cross-fitted, so honest)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import nsgain as N
import sys; sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
B = load(3); sid = B['ids']; W = np.load(_repo_io('wacf-train.npz')); pos = {str(i): k for k, i in enumerate(W['ids'])}; P = W['prof'][[pos[i] for i in sid]]
lag = [6, 8, 10, 13, 16, 20, 25]
Xa = np.c_[np.log(np.clip(P[:, lag], 1e-3, None) / np.clip(P[:, [4]], 1e-3, None)), np.log(np.clip(P[:, 4], 1e-3, None)), np.log(np.clip(P[:, 0] - P[:, 4], 1e-3, None))]
y, c61, nz, fail = N.y, N.c61, N.nz, N.fail[0]
R = (y - c61)   # [3, 400]
bands = {'clean<11': (nz < 11) & ~fail, 'noisy>=12': (nz >= 12) & ~fail, 'all': ~fail}
for nm, m in bands.items():
    idx = np.where(m)[0]; r2s, gains = [], []
    for k in range(3):
        for rep in range(5):
            pr = np.zeros(len(idx))
            for a, b in KFold(10, shuffle=True, random_state=rep).split(idx):
                mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 21))).fit(Xa[idx[a]], R[k, idx[a]]); pr[b] = mdl.predict(Xa[idx[b]])
            r = R[k, idx]; r2s.append(1 - ((r - pr) ** 2).sum() / ((r - r.mean()) ** 2).sum()); gains.append((r ** 2).mean() - ((r - pr) ** 2).mean())
    print(f'{nm:10s} n {len(idx)}: CV R2 {np.mean(r2s):+.4f} (sd {np.std(r2s):.4f})  band dMSE {-np.mean(gains):+.3f}')
# control: same pipeline with rows' ACF features permuted (null distribution of the CV R2 for this procedure)
rng = np.random.RandomState(1); m = bands['noisy>=12']; idx = np.where(m)[0]; nulls = []
for t in range(40):
    Xp = Xa.copy(); Xp[idx] = Xa[rng.permutation(idx)]; pr = np.zeros(len(idx)); k = t % 3
    for a, b in KFold(10, shuffle=True, random_state=t).split(idx):
        pr[b] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 21))).fit(Xp[idx[a]], R[k, idx[a]]).predict(Xp[idx[b]])
    r = R[k, idx]; nulls.append(1 - ((r - pr) ** 2).sum() / ((r - r.mean()) ** 2).sum())
print('noisy null CV R2: mean %.4f, 95%% %.4f' % (np.mean(nulls), np.quantile(nulls, 0.95)))
