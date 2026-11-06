# push-1300 variant of push-1200/synaug_ridge2.py reading the u1b (boundary-free-trained U-Net) feature files
"""Sim-augmented Ridge partners as 053 components (dev, Study900 id order): RidgeCV (production: u1dmavgH2Z-anghcor 98 cols, alpha grid) and
log-target Ridge alpha 2 (production: geometric mean over the 4 variants), fitted on real training rows [+ degraded copies of clean train images
whose source is not in the validation fold]. FEAT = H2Z (93 cols, synth rows available now) or anghcor (98 cols, needs synthrows-<set>-<v>-anghcor).
usage: synaug_ridge2.py FEAT TAGS(comma, or none) OUT"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
ALPHAS = np.array([float(a) for a in __import__('os').environ.get('ALPHAS', '0.1,1,10,100,1000').split(',')])   # default = Study900 ridge/run.py grid
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
P = Path(_repo_io('../..')); lab = {r.split(',')[0]: float(r.split(',')[1]) for r in open(_repo_path(P, 'data/train.csv')).read().split()[1:]}
feat, tags, out = sys.argv[1], ([] if sys.argv[2] == 'none' else sys.argv[2].split(',')), sys.argv[3]
TRMIN = float(sys.argv[4]) if len(sys.argv) > 4 else -1.0   # only real training rows with pixel noise >= TRMIN (gray) enter the fit
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'),
          '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
B = load(3); sid = B['ids']
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
def files(v):
    if feat == 'H2Z': return f'../push-500/feat-{v}-train.npz', [f'../push-500/synthrows-train_{t}-{v}.npz' for t in tags]
    return f'feat-{v}-anghcorB-train.npz', [f'synthrows-{t}-{v}-anghcorB.npz' for t in tags]
KEYC = ['u_ph', 'z_f25', 'sh_cor15', 'het4_u', 'sizefield_std_s32', 'm_la_mean', 'm_el_pix', 'n_obj', 'ang_c4', 'u_ph_hard']
def expand(X, keys):   # EXPAND=1: append squares of key columns and their products with the pixel-noise column (noise-dependent attenuation)
    if __import__('os').environ.get('EXPAND') != '1': return X
    J = [keys.index(k) for k in KEYC]; nz = X[:, keys.index('noise')][:, None]; Z = X[:, J]
    return np.c_[X, Z ** 2, Z * nz]
def oof(v, model):
    fr, fs = files(v); F = np.load(_repo_io(fr), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
    SX, SY, SID = [], [], []
    for f in fs:
        S = np.load(_repo_io(f), allow_pickle=True); SX.append(S['X'].astype(float)); SID += [str(i) for i in S['ids']]; SY += [lab[str(i)] for i in S['ids']]
    SX = np.vstack(SX) if SX else np.zeros((0, XA.shape[1])); SY = np.array(SY); SID = np.array(SID); O = np.zeros((3, 400))
    XA = expand(XA, keys); SX = expand(SX, keys) if len(SX) else SX
    NMIN = float(__import__('os').environ.get('NOISEMIN', '-1'))
    if NMIN > 0 and len(SX):
        _k = [str(x) for x in np.load(_repo_io(files(v)[1][0]), allow_pickle=True)['keys']]; _nz = SX[:, _k.index('noise')] * 255; _m = (_nz >= NMIN) & (_nz < float(__import__('os').environ.get('NOISEMAX', '1e9'))); SX, SY, SID = SX[_m], SY[_m], SID[_m]
    if __import__('os').environ.get('PERM'): SY = SY[np.random.RandomState(int(__import__('os').environ['PERM'])).permutation(len(SY))]   # null control: shuffled sim labels
    for p_, (name, path) in enumerate(splits.items()):
        d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; g = np.zeros(len(y))
        for k in range(5):
            tr, va = fold != k, fold == k; tr = tr & (np.nan_to_num(X[:, keys.index('noise')] * 255, nan=0) >= TRMIN); vids = set(np.array(ids)[va]); keep = np.array([s not in vids for s in SID], bool) if len(SID) else np.zeros(0, bool)
            SXk = SX[keep]
            LAM = float(__import__('os').environ.get('SHIFTFIX', '0'))   # SHIFTFIX=lam: add lam*(real noisy-clean shift - paired copy-source shift) to copies (train-fold rows only)
            if LAM > 0 and len(SXk):
                nzt = np.nan_to_num(X[:, keys.index('noise')] * 255, nan=0); Cm, Nm = tr & (nzt < 9.5), tr & (nzt >= 12)
                real = np.nanmean(X[Nm], 0) - np.nanmean(X[Cm], 0); sim = np.nanmean(SXk - XA[[pos[s] for s in SID[keep]]], 0)
                fix = np.nan_to_num(real - sim); fix[keys.index('noise')] = 0; SXk = SXk + LAM * fix
            Xt = np.vstack([X[tr], SXk]); yt = np.r_[y[tr], SY[keep]]
            SW = float(__import__('os').environ.get('SIMW', '0')); w = np.r_[np.ones(tr.sum()), np.full(keep.sum(), SW * tr.sum() / max(keep.sum(), 1) if SW > 0 else 1.0)]   # SIMW=m: sims' total weight = m x real rows
            if model == 'ridge': m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(Xt, yt, ridgecv__sample_weight=w); g[va] = m.predict(X[va])
            else:
                LA = __import__('os').environ.get('LOGALPHAS')
                m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=[float(a) for a in LA.split(',')]) if LA else Ridge(alpha=2.0)); m.fit(Xt, np.log(yt), **{m.steps[-1][0] + '__sample_weight': w}); g[va] = m.predict(X[va])
        pp = {i: j for j, i in enumerate(ids)}; O[p_] = g[[pp[i] for i in sid]]
    return O
R4 = __import__('os').environ.get('RIDGE4') == '1'   # RIDGE4=1: RidgeCV averaged over the 4 variant sets (as log-Ridge)
ridge = np.mean([oof(v, 'ridge') for v in V], 0) if R4 else oof('u1dmavgH2Z', 'ridge'); logr = np.exp(np.mean([oof(v, 'log') for v in V], 0))
np.savez(_repo_io(f'{out}.npz'), ridge=ridge, logridge=logr, ids=np.array(sid)); print('saved', out)
