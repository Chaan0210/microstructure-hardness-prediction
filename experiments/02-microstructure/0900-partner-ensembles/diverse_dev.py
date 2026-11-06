"""Diversity partners for the 045 grain block, Study900 protocol: the 400 dev ids, the three saved 5-fold partitions, preprocessing (median imputation,
standardisation, optional transforms) and hyper-parameter choice inside each outer training fold only (inner CV or analytic LOO). Features: the
98-column average-variant matrix (push-800/eda/resid045.npz, as in Study900) unless stated. For each model: standalone RMSE, error correlation with
the 045 grain mean, 045-structure RMSE with the partner taking share a of the grain block (0.2*021 + 0.8*((1-a)*g + a*m)), dev-implied optimal share.
usage: diverse_dev.py NAME[,NAME...]   -> diverse-<NAME>.npz (OOF [3, 400])"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, SplineTransformer
from sklearn.linear_model import RidgeCV, HuberRegressor, ElasticNetCV, LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
D = np.load(_repo_path(E, 'probabilistic-grains-900/ridge/predictions.npz'), allow_pickle=True)
ids = [str(i) for i in D['ids']]; Y, FOLD, P021, G, FAIL, RIDGE = D['y'], D['fold'], D['p021'], D['grain045'], D['failed'], D['ridge']
R = np.load(_repo_path(E, 'push-800/eda/resid045.npz'), allow_pickle=True); rpos = {str(i): k for k, i in enumerate(R['ids'])}; X = R['X'][[rpos[i] for i in ids]].astype(float)
ALPHAS = np.logspace(-2, 4, 25); inner = KFold(5, shuffle=True, random_state=0)
def base(): return [SimpleImputer(strategy='median'), StandardScaler()]
MODELS = {
    'ridge': lambda: make_pipeline(*base(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])),
    'ridge_fine': lambda: make_pipeline(*base(), RidgeCV(alphas=ALPHAS)),
    'ridge_qt': lambda: make_pipeline(SimpleImputer(strategy='median'), QuantileTransformer(n_quantiles=100, output_distribution='normal'), StandardScaler(), RidgeCV(alphas=ALPHAS)),
    'huber': lambda: make_pipeline(*base(), GridSearchCV(HuberRegressor(max_iter=2000), {'alpha': [1e-3, 1e-2, 0.1, 1, 10], 'epsilon': [1.35, 2.0]}, cv=inner, scoring='neg_mean_squared_error')),
    'enet': lambda: make_pipeline(*base(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=inner, max_iter=20000)),
    'ridge_a10': lambda: make_pipeline(*base(), RidgeCV(alphas=[10.0])),
    'ridge_a1': lambda: make_pipeline(*base(), RidgeCV(alphas=[1.0])),
    'ridge_a03': lambda: make_pipeline(*base(), RidgeCV(alphas=[0.3])),
    'ridge_a01': lambda: make_pipeline(*base(), RidgeCV(alphas=[0.1])),
    'ridge_a001': lambda: make_pipeline(*base(), RidgeCV(alphas=[0.01])),
    'ridge_a3': lambda: make_pipeline(*base(), RidgeCV(alphas=[3.0])),
    'ridge_a2': lambda: make_pipeline(*base(), RidgeCV(alphas=[2.0])),
    'ridge_a30': lambda: make_pipeline(*base(), RidgeCV(alphas=[30.0])),
    'ridge_a100': lambda: make_pipeline(*base(), RidgeCV(alphas=[100.0])),
    'pls': lambda: make_pipeline(*base(), GridSearchCV(PLSRegression(scale=False), {'n_components': [2, 4, 6, 8, 12, 16]}, cv=inner, scoring='neg_mean_squared_error')),
    'spline_ridge': lambda: make_pipeline(*base(), SplineTransformer(n_knots=4, degree=2), StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 21))),
    'krr_rbf': lambda: make_pipeline(*base(), GridSearchCV(KernelRidge(kernel='rbf'), {'alpha': [0.03, 0.1, 0.3, 1], 'gamma': [0.001, 0.003, 0.01]}, cv=inner, scoring='neg_mean_squared_error')),
    'svr_rbf': lambda: make_pipeline(*base(), GridSearchCV(SVR(kernel='rbf'), {'C': [10, 30, 100, 300], 'gamma': [0.001, 0.003, 0.01], 'epsilon': [1, 3]}, cv=inner, scoring='neg_mean_squared_error')),
    'gp_lin_rbf': lambda: make_pipeline(*base(), GaussianProcessRegressor(ConstantKernel(1.0) * DotProduct(1.0) + ConstantKernel(1.0) * RBF(10.0) + WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=1, random_state=0)),
}
VSETS = {v: (lambda F: F['X'][[[str(i) for i in F['ids']].index(j) for j in ids]].astype(float))(np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True))
         for v in ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z')}
PHYS = ['u_ph', 'u_ph_hard', 'm_ph_hard', 'aw_inv_sqrt_d', 'grain_inv_sqrt_d', 'het4_u', 'het4_u_mx', 'pore_frac', 'px_pore_frac', 'ar_wmean', 'orient_coherence',
        'ang_c4', 'ang_s4', 'n_obj', 'grain_logA_std', 'sh_cor15', 'z_f25', 'noise', 'm_el_pix', 'sp_contiguity']
KEYS = [str(k) for k in R['keys']]; PIDX = [KEYS.index(c) for c in PHYS]
CORE = ['u_ph', 'm_ph_hard', 'aw_inv_sqrt_d', 'grain_inv_sqrt_d', 'het4_u', 'pore_frac', 'ar_wmean', 'orient_coherence', 'ang_c4', 'n_obj', 'sh_cor15', 'noise']
CIDX = [KEYS.index(c) for c in CORE]
def interact(XX):   # 98 columns + pairwise products and squares of the 12 core columns (standardised with train statistics inside the pipeline later)
    C = XX[:, CIDX]; C = np.where(np.isnan(C), np.nanmedian(C, 0), C); C = (C - np.nanmean(C, 0)) / (np.nanstd(C, 0) + 1e-9)
    pr = [C[:, i] * C[:, j] for i in range(len(CIDX)) for j in range(i, len(CIDX))]; return np.c_[XX, np.array(pr).T]
WD = np.load(_repo_path(H, 'wide-train.npz'), allow_pickle=True); WX = WD['X'][[[str(i) for i in WD['ids']].index(j) for j in ids]].astype(float)
ZS = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); ZX = ZS['X'][[[str(i) for i in ZS['ids']].index(j) for j in ids]].astype(float)   # zero-shot masks for every row
def oof(name):
    phys = '_phys' in name; wide = '_wide' in name; inter = '_int' in name; zs = '_zs' in name; zonly = '_zonly' in name; name = name.replace('_phys', '').replace('_wide', '').replace('_int', '').replace('_zonly', '').replace('_zs', '')
    out = np.zeros_like(Y); var4 = name.endswith('_var4'); logy = '_logy' in name; mname = name.replace('_var4', '').replace('_logy', '')
    for k in range(3):
        for f in range(5):
            tr, va = FOLD[k] != f, FOLD[k] == f; yt = np.log(Y[k][tr]) if logy else Y[k][tr]; ps = []
            for XX in (VSETS.values() if var4 else [ZX if zonly else (WX if wide else X)]):
                XX = XX[:, PIDX] if phys else XX
                XX = interact(XX) if inter else XX
                XX = np.c_[XX, ZX] if zs else XX
                m = MODELS[mname](); m.fit(XX[tr], yt); ps.append(np.ravel(m.predict(XX[va])))
            pr = np.mean(ps, 0); out[k][va] = np.exp(pr) if logy else pr
    return out
def s045(m, a, W=0.8):
    h = (1 - a) * G + a * m; q = np.where(FAIL, P021, (1 - W) * P021 + W * h); return np.sqrt(((q - Y) ** 2).mean())
base45 = s045(G, 0.0)
for name in sys.argv[1].split(','):
    f = _repo_path(H, f'diverse-{name}.npz')
    M = np.load(_repo_io(f))['oof'] if f.exists() else oof(name)
    if not f.exists(): np.savez(_repo_io(f), oof=M, ids=np.array(ids))
    Pb = np.where(FAIL, P021, 0.2 * P021 + 0.8 * G); Dd = np.where(FAIL, 0, M - G); lam = -np.mean(Dd * (Pb - Y)) / np.mean(Dd ** 2)
    print(f'{name:13s} alone {np.sqrt(((M - Y) ** 2).mean()):.3f}  errcorr {np.corrcoef((M - Y).ravel(), (G - Y).ravel())[0, 1]:.3f}  '
          + ' '.join(f'a{a:.1f} {base45 - s045(M, a):+.4f}' for a in (0.2, 0.4)) + f'  opt total share {lam:.3f}'
          + '  per-part(a0.4) ' + ' '.join(f'{np.sqrt(((np.where(FAIL[k], P021[k], 0.2 * P021[k] + 0.8 * G[k]) - Y[k]) ** 2).mean()) - np.sqrt(((np.where(FAIL[k], P021[k], 0.2 * P021[k] + 0.8 * (0.6 * G[k] + 0.4 * M[k])) - Y[k]) ** 2).mean()):+.3f}' for k in range(3)), flush=True)
