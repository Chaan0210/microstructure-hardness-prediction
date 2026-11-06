"""All linear partners of candidate 076 in one place, dev OOF (Study900 400 rows x 3 partitions) and final (500 train rows -> 1000 test):
  plain (used on non-gated rows, 'nb' = without het+intensity groups; 'full' kept for gated gmm): RidgeCV u1dmavgH2Z-anghcor, logRidge a2 x 4 variants
  (geometric mean), gmm-Ridge (push-900/gmm_partner.py on u1dmavgH2Z-anghcor + per-image GMM phase measurements);
  DTA arm B (gated rows): Ridge / logRidge with label-preserving degraded copies (measured noise >= 12) of the training sources, alpha by inner 5-fold
  CV grouped by source scored on real rows with noise >= 12 (push-1300/srcw.py arm B, final_srcwB.py).
usage: c14_partners.py EXPROOT SYNTHDIR SETS(comma) OUT.npz     (run from a push-1300 dir that holds abl/groups.json)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import GroupKFold
E, SD, SETS, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3].split(','), sys.argv[4]
P = E.parent; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io(P / 'data/train.csv')))}; order = [r['ID'] for r in csv.DictReader(open(_repo_io(P / 'data/sample_submission.csv')))]
sid = [str(i) for i in np.load(_repo_io(E / 'probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]; dev = set(sid)
splits = [E / 'resolution-216/ensemble-OOF.npz', E / 'resolution-repartition-217/20261031/ensemble-OOF.npz', E / 'resolution-repartition-217/20261101/ensemble-OOF.npz']
G = json.load(open(_repo_io('abl/groups.json'))); drop = set(G['intensity'].split() + G['het'].split()); V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
src = open(_repo_io(E / 'push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]; gp = {'__file__': str(E / 'push-900/gmm_partner.py'), '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), gp)
S9 = E / 'probabilistic-grains-900'
def feat(v, split):
    F = np.load(_repo_io(E / f'push-800/feat-{v}-anghcor-{split}.npz'), allow_pickle=True); return [str(k) for k in F['keys']], [str(i) for i in F['ids']], F['X'].astype(float)
def mk(kind, a=None):
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    return make_pipeline(imp, StandardScaler(), Ridge(alpha=a if a is not None else 2.0) if kind == 'log' else (Ridge(alpha=a) if a is not None else RidgeCV(alphas=[0.1, 1, 10, 100, 1000])))
def fp(kind, Xt, yt, Xv):
    if kind == 'gmm': return gp['fit_predict'](Xt, yt, Xv, 19001)[0]
    m = mk(kind).fit(Xt, np.log(yt) if kind == 'log' else yt); p = m.predict(Xv); return np.exp(p) if kind == 'log' else p
gmean = lambda a: np.exp(np.mean(np.log(a), 0))
out = {}
# ---------- plain partners
Btr = np.load(_repo_io(S9 / 'object-bank-train.npz'), allow_pickle=True); Bte = np.load(_repo_io(S9 / 'object-bank-test.npz'), allow_pickle=True)
for name, kind, vs in (('ridge', 'ridge', ['u1dmavgH2Z']), ('logridge', 'log', V), ('gmm', 'gmm', ['u1dmavgH2Z'])):
    for tag in ('full', 'nb'):
        dv, te = [], []
        for v in vs:
            keys, ids, X = feat(v, 'train'); _, eids, Xe = feat(v, 'test'); assert eids == order
            cols = [j for j, k in enumerate(keys) if tag == 'full' or k not in drop]; X, Xe = X[:, cols], Xe[:, cols]
            if kind == 'gmm': X = np.c_[X, gp['gmm_feats'](Btr, ids)]; Xe = np.c_[Xe, gp['gmm_feats'](Bte, order)]
            pos = {i: k for k, i in enumerate(ids)}; O = np.zeros((3, 400))
            for p_, path in enumerate(splits):
                d = np.load(_repo_io(path)); di, fold = [str(i) for i in d['ids']], d['fold']
                for k in range(5):
                    tr = [i for i, f in zip(di, fold) if f != k]; va = [i for i, f in zip(di, fold) if f == k]
                    O[p_, [sid.index(i) for i in va]] = fp(kind, X[[pos[i] for i in tr]], np.array([lab[i] for i in tr]), X[[pos[i] for i in va]])
            dv.append(O); te.append(fp(kind, X, np.array([lab[i] for i in ids]), Xe))
        agg = gmean if kind == 'log' else (lambda a: np.mean(a, 0))
        out[f'{name}_{tag}_dev'] = agg(np.array(dv)); out[f'{name}_{tag}_test'] = agg(np.array(te)); print(name, tag, flush=True)
# ---------- DTA arm B
GR = {'ridge': [0.1, 0.3, 1, 3, 10, 30, 100, 1000], 'log': [0.3, 1, 2, 3, 10, 30, 100]}
def pipe(kind, a): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True) if kind == 'ridge' else SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=a))
tf = lambda kind, y: np.log(y) if kind == 'log' else y; itf = lambda kind, p: np.exp(p) if kind == 'log' else p
def fitB(kind, Xr, yr, sr, nzr, Xc, yc, sc):
    X = np.vstack([Xr, Xc]); y = np.r_[yr, yc]; srcs = np.r_[sr, sc]; isreal = np.r_[np.ones(len(yr), bool), np.zeros(len(yc), bool)]; nzall = np.r_[nzr, np.full(len(yc), 99.0)]; best = None
    for a in GR[kind]:
        err = []
        for tr, va in GroupKFold(5).split(X, y, srcs):
            m = pipe(kind, a).fit(X[tr], tf(kind, y[tr])); sel = va[isreal[va] & (nzall[va] >= 12)]
            if sel.size: err += list((itf(kind, m.predict(X[sel])) - y[sel]) ** 2)
        e = np.mean(err)
        if best is None or e < best[0]: best = (e, a)
    return pipe(kind, best[1]).fit(X, tf(kind, y)), best[1]
if SETS != ['none']:
    for kind, vs in (('ridge', ['u1dmavgH2Z']), ('log', V)):
        O, T, al = [], [], []
        for v in vs:
            keys, ids, XA = feat(v, 'train'); _, eids, Xe = feat(v, 'test'); ni = keys.index('noise'); NZ = np.nan_to_num(XA[:, ni] * 255, nan=0); pos = {i: k for k, i in enumerate(ids)}
            SX, SID = [], []
            for s in SETS:
                S = np.load(_repo_io(SD / f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); assert [str(k) for k in S['keys']] == keys
                m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
            SX = np.vstack(SX); SID = np.array(SID); o = np.zeros((3, 400))
            for p_, path in enumerate(splits):
                d = np.load(_repo_io(path)); di, fold = [str(i) for i in d['ids']], d['fold']
                for k in range(5):
                    tr = [i for i, f in zip(di, fold) if f != k]; va = [i for i, f in zip(di, fold) if f == k]; vset = set(va); kp = np.array([s not in vset for s in SID]); jt = np.array([pos[i] for i in tr])
                    m, a = fitB(kind, XA[jt], np.array([lab[i] for i in tr]), np.array(tr), NZ[jt], SX[kp], np.array([lab[s] for s in SID[kp]]), SID[kp]); al.append(a)
                    o[p_, [sid.index(i) for i in va]] = itf(kind, m.predict(XA[[pos[i] for i in va]]))
            O.append(o); jt = np.array([pos[i] for i in ids])
            m, a = fitB(kind, XA[jt], np.array([lab[i] for i in ids]), np.array(ids), NZ[jt], SX, np.array([lab[s] for s in SID]), SID); T.append(itf(kind, m.predict(Xe))); out[f'alpha_final_{kind}_{v}'] = a
        key = 'ridge' if kind == 'ridge' else 'logridge'
        out[f'dta_{key}_dev'] = O[0] if kind == 'ridge' else gmean(np.array(O)); out[f'dta_{key}_test'] = T[0] if kind == 'ridge' else gmean(np.array(T))
        print('DTA', key, 'dev alphas median', np.median(al), 'n copies', len(SID), flush=True)
_, ids, X = feat('u1dmavgH2Z', 'test'); out['test_noise'] = np.nan_to_num(X[:, feat('u1dmavgH2Z', 'test')[0].index('noise')] * 255, nan=0)
np.savez(_repo_io(OUT), sid=np.array(sid), order=np.array(order), **out); print('saved', OUT)
