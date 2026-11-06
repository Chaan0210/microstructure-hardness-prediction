"""Heteroscedastic (noise-preset weighted) Ridge partners. Hypothesis (a priori, from the residual covariance of six model families):
noisy-preset rows (pixel noise >= 12 gray) carry ~2x the irreducible error of clean rows, so the shared linear hardness function should be fitted
with weights ~1/variance (clean 1, noisy w). Same Study900 folds/protocol as push-900/diverse_dev.py; the two Ridge slots of c054 are swapped for
weighted versions with an unweighted control computed by this same code.
usage: wridge_dev.py   -> wridge-<tag>.npz (OOF [3, 400]) + printed table"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P9 = _repo_path(E, 'push-900')
sys.path.insert(0, str(P9)); from blend_dev import load, rmse, parts
B = load(9); ids = B['ids']; Y = B['y']; FOLD = B['fold']
R = np.load(_repo_path(E, 'push-800/eda/resid045.npz'), allow_pickle=True); rpos = {str(i): k for k, i in enumerate(R['ids'])}; X = R['X'][[rpos[i] for i in ids]].astype(float)
KEYS = [str(k) for k in R['keys']]; NZ = np.nan_to_num(X[:, KEYS.index('noise')] * 255, nan=0.0)
VSETS = [(lambda F: F['X'][[[str(i) for i in F['ids']].index(j) for j in ids]].astype(float))(np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True))
         for v in ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z')]
def model(alphas): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=alphas))
def oof(kind, w):
    out = np.zeros((3, len(Y))); sw = np.where(NZ >= 12, w, 1.0)
    for k in range(3):
        for f in range(5):
            tr, va = FOLD[k] != f, FOLD[k] == f
            if kind == 'ridge':
                m = model([0.1, 1, 10, 100, 1000]); m.fit(X[tr], Y[tr], ridgecv__sample_weight=sw[tr]); out[k][va] = m.predict(X[va])
            else:   # log-target alpha 2, geometric mean over the 4 U-Net variant matrices (as push-900 ridge_a2_logy_var4)
                ps = []
                for XX in VSETS:
                    m = model([2.0]); m.fit(XX[tr], np.log(Y[tr]), ridgecv__sample_weight=sw[tr]); ps.append(m.predict(XX[va]))
                out[k][va] = np.exp(np.mean(ps, 0))
    return out
def c054(r, lr):
    c51 = 0.2 * B['p021'] + 0.48 * B['g'] + 0.16 * r + 0.16 * lr; return np.where(B['fail'], B['p021'], 0.77 * c51 + 0.08 * B['set'] + 0.15 * B['gmm'])
bands = {'clean': NZ < 9.5, 'mid': (NZ >= 9.5) & (NZ < 12), 'noisy': NZ >= 12}
def show(tag, c, ref):
    d = rmse(ref, Y) - rmse(c, Y); pp = parts(ref, Y) - parts(c, Y)
    bb = ' '.join(f'{n} {np.sqrt(((ref[:, m] - Y[m]) ** 2).mean()) - np.sqrt(((c[:, m] - Y[m]) ** 2).mean()):+.3f}' for n, m in bands.items())
    print(f'{tag:22s} {rmse(c, Y):.4f} ({d:+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in pp) + f'; {bb})', flush=True)
res = {}
for w in (1.0, 0.6, 0.4, 0.25):
    for kind in ('ridge', 'logridge'):
        f = _repo_path(H, f'wridge-{kind}-w{w}.npz')
        res[kind, w] = np.load(_repo_io(f))['oof'] if f.exists() else oof(kind, w)
        if not f.exists(): np.savez(_repo_io(f), oof=res[kind, w], ids=np.array(ids))
        print(f'{kind:8s} w{w}: alone {rmse(res[kind, w], Y):.3f}', ' '.join(f'{n} {np.sqrt(((res[kind, w][:, m] - Y[m]) ** 2).mean()):.2f}' for n, m in bands.items()), flush=True)
ref = c054(res['ridge', 1.0], res['logridge', 1.0])
print('stored c054', round(rmse(B['c054'], Y), 4), ' recomputed control', round(rmse(ref, Y), 4))
for w in (0.6, 0.4, 0.25):
    show(f'both w{w}', c054(res['ridge', w], res['logridge', w]), ref)
    show(f'ridge only w{w}', c054(res['ridge', w], res['logridge', 1.0]), ref)
    show(f'logridge only w{w}', c054(res['ridge', 1.0], res['logridge', w]), ref)
