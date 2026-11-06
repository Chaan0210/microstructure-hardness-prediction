"""Zone joint-information test (dev, one partition at a time). Outer folds as official dev; base = 039 avg-variant grain OOF (3 seeds, existing file).
Inside each outer-training set: inner 5-fold OOF of TabPFN-3.5 (1 seed) on the same angh features -> residuals r_in. Residual models (RidgeCV, GCV inside training rows):
  global   : 97 angh global features (stacking control, no new info)
  low_real : per-image mean/std of 6 zone channels + 15 cross-zone channel correlations (27 cols) from real zone vectors
  low_shuf : same with phase channels (zf25, zf50) permuted jointly across zones within each image
  setk_real: random-Fourier mean embedding (set kernel, RBF, D=512, median-heuristic bandwidth) of standardized zone vectors
  setk_shuf: same on shuffled zone vectors
Corrected grain = outer OOF + predicted residual; fixed W = 0.75 blend with best021. usage: zone_joint.py PARTITION"""
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
F = np.load(_repo_io('feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); fp = {str(i): k for k, i in enumerate(F['ids'])}; keys = [str(k) for k in F['keys']]
assert all(i in fp for i in ids); X = F['X'][[fp[i] for i in ids]].astype(float)
ZV = np.load(_repo_io('zonevec-train.npz'), allow_pickle=True); zp = {str(i): k for k, i in enumerate(ZV['ids'])}; Zr = ZV['Z'][[zp[i] for i in ids]].astype(np.float64)
rng = np.random.RandomState(777); Zs = Zr.copy()
for n in range(len(ids)): perm = rng.permutation(16); Zs[n, :, 4:6] = Zr[n, perm, 4:6]
def lowstats(Z):
    mu = Z.mean(1); sd = Z.std(1); cc = []
    for a in range(6):
        for b in range(a + 1, 6):
            za = Z[:, :, a] - Z[:, :, a].mean(1, keepdims=True); zb = Z[:, :, b] - Z[:, :, b].mean(1, keepdims=True)
            cc.append((za * zb).mean(1) / (za.std(1) * zb.std(1) + 1e-9))
    return np.c_[mu, sd, np.array(cc).T]
L_real, L_shuf = lowstats(Zr), lowstats(Zs)
def setk(Ztr, Zall, seed=0, Dm=512):
    m = Ztr.reshape(-1, 6).mean(0); s = Ztr.reshape(-1, 6).std(0) + 1e-9
    A = (Zall - m) / s; At = (Ztr - m) / s
    r = np.random.RandomState(seed); flat = At.reshape(-1, 6); i1 = r.randint(len(flat), size=3000); i2 = r.randint(len(flat), size=3000)
    sig = np.median(np.linalg.norm(flat[i1] - flat[i2], axis=1))
    Wm = r.normal(0, 1 / sig, (6, Dm)); b = r.uniform(0, 2 * np.pi, Dm)
    return np.sqrt(2 / Dm) * np.cos(A @ Wm + b).mean(1)
def ridge_pred(Ftr, rtr, Fva):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    m = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit((Ftr - mu) / sd, rtr); return m.predict((Fva - mu) / sd), m.alpha_
arms = ['base', 'global', 'low_real', 'low_shuf', 'setk_real', 'setk_shuf']
G = {a: g_out.copy() for a in arms}; alphas = {a: [] for a in arms[1:]}
for k in range(5):
    tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
    fi = np.arange(len(tr)) % 5; np.random.RandomState(1000 + k).shuffle(fi); g_in = np.zeros(len(tr))
    for j in range(5):
        m = harness.tabpfn_new('v3.5', 100 * 0 + k, 'mps'); m.fit(X[tr][fi != j], y[tr][fi != j]); g_in[fi == j] = m.predict(X[tr][fi == j])
    r_in = y[tr] - g_in
    feats = {'global': np.nan_to_num(X), 'low_real': L_real, 'low_shuf': L_shuf}
    feats['setk_real'] = setk(Zr[tr], Zr, seed=k); feats['setk_shuf'] = setk(Zs[tr], Zs, seed=k)
    for a in arms[1:]:
        d, al = ridge_pred(feats[a][tr], r_in, feats[a][va]); G[a][va] = g_out[va] + d; alphas[a].append(round(float(al), 2))
    print('fold', k, 'inner resid rmse %.3f' % np.sqrt(np.mean(r_in ** 2)), {a: alphas[a][-1] for a in arms[1:]}, flush=True)
def blend(g): return 0.25 * p021 + 0.75 * np.where(failed, p021, g)
se = {a: (blend(G[a]) - y) ** 2 for a in arms}
nz = X[:, keys.index('noise')] * 255; el = X[:, keys.index('m_el_glob')]
print('partition', part)
for a in arms:
    d = se[a] - se['base']; per = [np.mean(d[fold == k]) for k in range(5)]
    print('%-10s RMSE %.4f | dMSE %+.2f (se %.2f) folds better %d/5 | LQ %.3f elong>0.7 %.3f | top5 rows share of gain %s' % (
        a, np.sqrt(se[a].mean()), d.mean(), d.std() / np.sqrt(len(d)), sum(p < 0 for p in per), np.sqrt(se[a][nz >= 11.9].mean()), np.sqrt(se[a][el > 0.7].mean()),
        ('%.2f' % (np.sort(d)[:5].sum() / d.sum()) if d.sum() < 0 else 'n/a')))
np.savez(_repo_io(f'zonejoint-{part}.npz'), **{a: G[a] for a in arms}, ids=np.array(ids))
