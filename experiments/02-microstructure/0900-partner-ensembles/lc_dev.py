"""Learning-curve check: do the Ridge partners gain more from extra training rows than TabPFN (dev models see 320 rows, final models 500)?
Same 400 dev ids and labels; K-fold with K in {3, 10} (seeded per partition; K=5 = the saved folds). Models: TabPFN-3.5 on the 044 sets
(4 variants x 3 seeds, as dev600), Ridge (Study900: RidgeCV alphas 0.1..1000 on the average-variant set), logRidge (alpha 2, log target, 4-variant
geometric mean). Reports RMSE per K and the optimal Ridge share inside the grain block (1-a)*TabPFN + a*partner.
usage: lc_dev.py K [K ...]   (TabPFN OOF cached in lc-<K>.npz)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, _repo_io('../push-300')); import harness
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
D = np.load(_repo_path(E, 'probabilistic-grains-900/ridge/predictions.npz'), allow_pickle=True); ids = [str(i) for i in D['ids']]; Y = D['y']
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
XV = {}
for v in V:
    F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); pos = {str(i): k for k, i in enumerate(F['ids'])}; XV[v] = F['X'][[pos[i] for i in ids]].astype(float)
y = Y[0]
def folds(K, k):
    if K == 5: return D['fold'][k]
    f = np.zeros(400, int)
    for j, (_, va) in enumerate(KFold(K, shuffle=True, random_state=7000 + 31 * k + K).split(np.arange(400))): f[va] = j
    return f
def oof_models(K):
    cache = _repo_path(H, f'lc-{K}.npz')
    T = np.load(_repo_io(cache))['tab'] if cache.exists() else None
    tab = np.zeros((3, 400)) if T is None else T; rid = np.zeros((3, 400)); lrid = np.zeros((3, 400))
    for k in range(3):
        fo = folds(K, k)
        for f in range(K):
            tr, va = fo != f, fo == f
            rid[k, va] = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(XV['u1dmavgH2Z'][tr], y[tr]).predict(XV['u1dmavgH2Z'][va])
            lrid[k, va] = np.exp(np.mean([make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(XV[v][tr], np.log(y[tr])).predict(XV[v][va]) for v in V], 0))
            if T is None:
                ps = []
                for v in V:
                    for sd in range(3):
                        m = harness.tabpfn_new('v3.5', 100 * sd + f, 'mps'); m.fit(XV[v][tr], y[tr]); ps.append(m.predict(XV[v][va]))
                tab[k, va] = np.mean(ps, 0)
        print(f'K={K} partition {k} done', flush=True)
    if T is None: np.savez(_repo_io(cache), tab=tab)
    return tab, rid, lrid
rm = lambda p: np.sqrt(((p - y) ** 2).mean())
for K in [int(a) for a in sys.argv[1:]]:
    if K == 5:
        Dd = [np.load(_repo_path(E, f'push-800/dev-{v}-anghcor.npz'), allow_pickle=True) for v in V]; parts = ['original', '20261031', '20261101']; tab = np.zeros((3, 400))
        for k, n in enumerate(parts):
            ii = [str(i) for i in Dd[0][f'{n}_ids']]; jj = [ii.index(i) for i in ids]; tab[k] = np.mean([d[f'{n}_g'] for d in Dd], 0)[jj]
        rid = D['ridge']; lrid = np.load(_repo_path(H, 'diverse-ridge_a2_logy_var4.npz'))['oof']
    else:
        tab, rid, lrid = oof_models(K)
    out = [f'K={K:2d} (train {int(400 * (K - 1) / K)}): TabPFN {rm(tab):.3f}  Ridge {rm(rid):.3f}  logRidge {rm(lrid):.3f}']
    for nm, M in (('Ridge', rid), ('logRidge', lrid)):
        d = M - tab; a = -np.mean(d * (tab - y)) / np.mean(d * d); out.append(f'opt share {nm} {a:.3f} (blend {rm(tab + a * d):.3f})')
    Cm = np.array([[np.mean((rid - tab) ** 2), np.mean((rid - tab) * (lrid - tab))], [np.mean((rid - tab) * (lrid - tab)), np.mean((lrid - tab) ** 2)]])
    b = np.array([np.mean((rid - tab) * (tab - y)), np.mean((lrid - tab) * (tab - y))]); w = -np.linalg.solve(Cm, b); out.append(f'joint (Ridge, logRidge) {w.round(3)}')
    print('  '.join(out), flush=True)
