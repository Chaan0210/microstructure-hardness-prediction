"""Realism-weighted degradation-transfer augmentation (RW-DTA), dev. Per outer fold: logistic domain classifier on label-free pixel statistics
(pixstats.npz) separating the outer-training real noisy rows (noise >= 12) from the degraded copies (sources outside the validation fold);
copy weight = density ratio p/(1-p), normalised to mean 1 over the copies, clipped at CLIP. Partners = production RidgeCV (u1dmavgH2Z) and
log-Ridge alpha 2 (geometric mean over 4 variants) with these sample weights. usage: rw_dta.py SETS OUTTAG  env CLIP (10), POW (1)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, csv, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200'))
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; SETS = sys.argv[1].split(','); tag = sys.argv[2]
CLIP = float(os.environ.get('CLIP', '10')); POW = float(os.environ.get('POW', '1'))
PS = np.load(_repo_io('pixstats.npz')); ptr = dict(zip([str(i) for i in PS['train_ids']], PS['train_X']))
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'),
          '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
def data(v):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
    SX, SID, SP = [], [], []
    for s in SETS:
        S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12
        pmap = dict(zip([str(i) for i in PS[f'{s}_ids']], PS[f'{s}_X'])); ii = [str(i) for i in np.array(S['ids'])[m]]
        SX.append(S['X'].astype(float)[m]); SID += ii; SP += [pmap[i] for i in ii]
    return [str(i) for i in F['ids']], F['X'].astype(float), ni, np.vstack(SX), np.array(SID), np.array(SP)
def weights(tr_ids, nz_tr, SPk):
    R = np.array([ptr[i] for i, n in zip(tr_ids, nz_tr) if n >= 12]); Z = np.vstack([R, SPk]); t = np.r_[np.ones(len(R)), np.zeros(len(SPk))]
    sc = StandardScaler().fit(Z); clf = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced').fit(sc.transform(Z), t)
    p = clf.predict_proba(sc.transform(SPk))[:, 1]; w = (p / (1 - p)) ** POW; w = w / w.mean(); return np.minimum(w, CLIP)
def oof(v, kind, W=None):
    ids_all, XA, ni, SX, SID, SP = data(v); pos = {i: k for k, i in enumerate(ids_all)}; O = np.zeros((3, 400)); Wout = {}
    for p_, (name, path) in enumerate(splits.items()):
        d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va); keep = np.array([s not in vs for s in SID])
            nz_tr = np.nan_to_num(XA[[pos[i] for i in tr], ni] * 255, nan=0)
            w = W[(p_, k)] if W is not None else weights(tr, nz_tr, SP[keep]); Wout[(p_, k)] = w
            X = np.vstack([XA[[pos[i] for i in tr]], SX[keep]]); yy = np.r_[[lab[i] for i in tr], [lab[s] for s in SID[keep]]]; sw = np.r_[np.ones(len(tr)), w]
            if kind == 'ridge': m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(X, yy, ridgecv__sample_weight=sw); pr = m.predict(XA[[pos[i] for i in va]])
            else: m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(X, np.log(yy), ridge__sample_weight=sw); pr = np.exp(m.predict(XA[[pos[i] for i in va]]))
            O[p_, [sid.index(i) for i in va]] = pr
    return O, Wout
r, W = oof('u1dmavgH2Z', 'ridge'); ls = [np.log(oof(v, 'log', W)[0]) for v in V]   # same weights for all variants (copies share images across variants)
ws = np.concatenate(list(W.values())); print(tag, 'weights: ESS/n', round(ws.sum() ** 2 / (ws ** 2).sum() / len(ws), 3), 'quantiles', np.round(np.quantile(ws, [.1, .5, .9, .99]), 2), flush=True)
np.savez(_repo_io(f'rw-{tag}.npz'), ridge=r, logridge=np.exp(np.mean(ls, 0)))
