"""Generic inner-OOF residual test on dev (base = 039 avg-variant grain OOF). Residual readers: RidgeCV (GCV within outer-training rows),
optional in-fold PCA. Arms: sem (16), pen (64), pen_pca10, rand64 (64 unselected N(0,1) columns: null control). usage: resid_test.py PARTITION"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.linear_model import RidgeCV
part = sys.argv[1]
D = np.load(_repo_io('../push-600/dev-u1dmavgH2Z-angh.npz'), allow_pickle=True)
ids = [str(i) for i in D[part + '_ids']]; y = D[part + '_y']; fold = D[part + '_fold']; p021 = D[part + '_p021']; failed = D[part + '_failed']; g_out = D[part + '_g']
F = np.load(_repo_io('feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); fp = {str(i): k for k, i in enumerate(F['ids'])}; keys = [str(k) for k in F['keys']]; X = F['X'][[fp[i] for i in ids]].astype(float)
P = np.load(_repo_io('penfeat-train.npz'), allow_pickle=True); pp = {str(i): k for k, i in enumerate(P['ids'])}; j = [pp[i] for i in ids]
arms = {'sem': P['SEM'][j], 'pen': P['PEN'][j], 'pen_pca10': P['PEN'][j], 'rand64': np.random.RandomState(9).normal(size=(len(ids), 64))}
G = {a: g_out.copy() for a in ['base'] + list(arms)}
for k in range(5):
    tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
    fi = np.arange(len(tr)) % 5; np.random.RandomState(1000 + k).shuffle(fi); g_in = np.zeros(len(tr))
    for jj in range(5):
        m = harness.tabpfn_new('v3.5', k, 'mps'); m.fit(X[tr][fi != jj], y[tr][fi != jj]); g_in[fi == jj] = m.predict(X[tr][fi == jj])
    r_in = y[tr] - g_in; line = []
    for a, Fm in arms.items():
        mu, sd = Fm[tr].mean(0), Fm[tr].std(0) + 1e-9; A_ = (Fm[tr] - mu) / sd; B_ = (Fm[va] - mu) / sd
        if a.endswith('pca10'):
            U, S, Vt = np.linalg.svd(A_, full_matrices=False); A_ = A_ @ Vt[:10].T; B_ = B_ @ Vt[:10].T
        rr = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(A_, r_in); G[a][va] = g_out[va] + rr.predict(B_); line.append('%s %.1f' % (a, rr.alpha_))
    print('fold', k, ' alphas:', ' '.join(line), flush=True)
bl = lambda g: 0.25 * p021 + 0.75 * np.where(failed, p021, g)
se = {a: (bl(G[a]) - y) ** 2 for a in G}; nz = X[:, keys.index('noise')] * 255; el = X[:, keys.index('m_el_glob')]
for a in G:
    d = se[a] - se['base']
    print('%-10s RMSE %.4f dMSE %+.2f (se %.2f) folds better %d/5 | LQ %.3f elong>0.7 %.3f' % (a, np.sqrt(se[a].mean()), d.mean(), d.std() / np.sqrt(len(d)), sum(np.mean(d[fold == k]) < 0 for k in range(5)), np.sqrt(se[a][nz >= 11.9].mean()), np.sqrt(se[a][el > 0.7].mean())))
