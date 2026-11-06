"""Plain linear partners (used on non-gated rows) without the het+intensity groups ('nb'): RidgeCV u1dmavgH2Z, logRidge a2 x 4 variants, gmm-Ridge.
dev: Study900 folds (OOF on all 400 rows; only non-gated rows are used in the blend). holdout: fit on dev-400, predict holdout-100."""
import csv, json, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
sys.path.insert(0, '../push-900'); from blend_dev import load
P = Path('../..'); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(P / 'data/train.csv'))}
dev = set(sid); splits = {'original': P / 'experiments/resolution-216/ensemble-OOF.npz', '20261031': P / 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz', '20261101': P / 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz'}
G = json.load(open('abl/groups.json')); drop = set(G['intensity'].split() + G['het'].split()); V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
src = open('../push-900/gmm_partner.py').read(); src = src[:src.index("mode = sys.argv[1]")]; gp = {'__file__': '../push-900/gmm_partner.py', '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), gp)
def mk(kind): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
def fp(kind, Xt, yt, Xv):
    if kind == 'gmm': return gp['fit_predict'](Xt, yt, Xv, 19001)[0]
    m = mk(kind).fit(Xt, np.log(yt) if kind == 'log' else yt); p = m.predict(Xv); return np.exp(p) if kind == 'log' else p
def feats(v, gmm=False):
    F = np.load(f'../push-800/feat-{v}-anghcor-train.npz', allow_pickle=True); keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]; X = F['X'].astype(float)
    keep = [j for j, k in enumerate(keys) if k not in drop]
    if gmm: Xg = gp['gmm_feats'](np.load('../probabilistic-grains-900/object-bank-train.npz', allow_pickle=True), ids); return ids, np.c_[X, Xg], np.c_[X[:, keep], Xg]
    return ids, X, X[:, keep]
out = {}
for name, kind, vs, gm in (('ridge', 'ridge', ['u1dmavgH2Z'], False), ('logridge', 'log', V, False), ('gmm', 'gmm', ['u1dmavgH2Z'], True)):
    for tag in ('full', 'nb'):
        dv, ho = [], []
        for v in vs:
            ids, Xf, Xn = feats(v, gm); X = Xf if tag == 'full' else Xn; pos = {i: k for k, i in enumerate(ids)}; O = np.zeros((3, 400))
            for p_, path in enumerate(splits.values()):
                d = np.load(path); di, fold = [str(i) for i in d['ids']], d['fold']
                for k in range(5):
                    tr = [i for i, f in zip(di, fold) if f != k]; va = [i for i, f in zip(di, fold) if f == k]
                    O[p_, [sid.index(i) for i in va]] = fp(kind, X[[pos[i] for i in tr]], np.array([lab[i] for i in tr]), X[[pos[i] for i in va]])
            trm = np.array([i in dev for i in ids]); dv.append(O); ho.append(fp(kind, X[trm], np.array([lab[i] for i in np.array(ids)[trm]]), X[~trm]))
            out['hid'] = np.array(ids)[~trm]
        gm_ = (lambda a: np.exp(np.mean(np.log(a), 0))) if kind == 'log' else (lambda a: np.mean(a, 0))
        out[f'{name}_{tag}_dev'] = gm_(np.array(dv)); out[f'{name}_{tag}_ho'] = gm_(np.array(ho)); print(name, tag, flush=True)
np.savez('nb-partners.npz', **out)
