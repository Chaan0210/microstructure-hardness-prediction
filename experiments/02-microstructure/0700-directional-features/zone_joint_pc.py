"""Positive control for the zone_joint.py procedure: base = 038 avg-variant (no angle), inner OOF on 038 features, residual model on the 2 angle columns
(+ the global set as a stacking control). If the procedure has power, 'angle' should reduce MSE (dev residual corr with ang_c4 was +0.13)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.linear_model import RidgeCV
part = sys.argv[1]
D = np.load(_repo_io('../push-300/dev-u1dmavgH2Z-v35.npz'), allow_pickle=True)
ids = [str(i) for i in D[part + '_ids']]; y = D[part + '_y']; fold = D[part + '_fold']; p021 = D[part + '_p021']; failed = D[part + '_failed']; g_out = D[part + '_g']
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); fp = {str(i): k for k, i in enumerate(F['ids'])}; X = F['X'][[fp[i] for i in ids]].astype(float)
A = np.load(_repo_io('angle-train.npz')); ap = {str(i): k for k, i in enumerate(A['ids'])}; ANG = A['X'][[ap[i] for i in ids]][:, :2]
G = {'base': g_out.copy(), 'angle': g_out.copy(), 'angle_x_rand': g_out.copy()}
rng = np.random.RandomState(5); RAND = rng.normal(size=(len(ids), 2))
for k in range(5):
    tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
    fi = np.arange(len(tr)) % 5; np.random.RandomState(1000 + k).shuffle(fi); g_in = np.zeros(len(tr))
    for j in range(5):
        m = harness.tabpfn_new('v3.5', k, 'mps'); m.fit(X[tr][fi != j], y[tr][fi != j]); g_in[fi == j] = m.predict(X[tr][fi == j])
    r_in = y[tr] - g_in
    for a, Fm in [('angle', ANG), ('angle_x_rand', RAND)]:
        mu, sd = Fm[tr].mean(0), Fm[tr].std(0); rr = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit((Fm[tr] - mu) / sd, r_in)
        G[a][va] = g_out[va] + rr.predict((Fm[va] - mu) / sd); print(k, a, 'alpha', round(float(rr.alpha_), 2), flush=True)
bl = lambda g: 0.25 * p021 + 0.75 * np.where(failed, p021, g)
se = {a: (bl(G[a]) - y) ** 2 for a in G}
for a in G:
    d = se[a] - se['base']; print('%-12s RMSE %.4f dMSE %+.2f (se %.2f) folds better %d/5' % (a, np.sqrt(se[a].mean()), d.mean(), d.std() / np.sqrt(len(d)), sum(np.mean(d[fold == k]) < 0 for k in range(5))))
