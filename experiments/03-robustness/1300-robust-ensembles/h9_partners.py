"""H9 columns (cumulant sigma-1 p34, contrast) added to the linear partners of 076: plain RidgeCV / logRidge(4 variants) / gmm-Ridge on non-gated rows
(nb columns + 2) and DTA Ridge / logRidge on gated rows (full columns + 2; the copies get their own cumulants? NO - copies have no cumulant measurement,
so for DTA partners the 2 columns are imputed by the training median for copies: tested both 'impute' and 'skip DTA'). Study900 dev folds; deltas applied
to the 076 dev blend (dev-combo.npy). Random-2-column control (same rng as the TabPFN screen)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
import nsgain as N
from robust_eval import report
P = Path(_repo_io('../..')); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
G = json.load(open(_repo_io('abl/groups.json'))); drop = set(G['intensity'].split() + G['het'].split()); V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
Cu = np.load(_repo_io('cumul-train.npz')); cpos = {str(i): j for j, i in enumerate(Cu['ids'])}; EX = {'phys': Cu['X'][:, [1, 4]], 'rand': np.random.default_rng(2026101509).normal(size=(500, 2))}
src = open(_repo_io('../push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]; gp = {'__file__': '../push-900/gmm_partner.py', '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), gp)
def mk(kind): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
def fp(kind, Xt, yt, Xv):
    if kind == 'gmm': return gp['fit_predict'](Xt, yt, Xv, 19001)[0]
    m = mk(kind).fit(Xt, np.log(yt) if kind == 'log' else yt); p = m.predict(Xv); return np.exp(p) if kind == 'log' else p
def oof(v, kind, ex, gmm=False):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]; pos = {i: j for j, i in enumerate(ids)}
    keep = [j for j, k in enumerate(keys) if k not in drop]; X = F['X'].astype(float)[:, keep]
    if gmm: X = np.c_[X, gp['gmm_feats'](np.load(_repo_io('../probabilistic-grains-900/object-bank-train.npz'), allow_pickle=True), ids)]
    if ex is not None: X = np.c_[X, ex[[cpos[i] for i in ids]]]
    O = np.zeros((3, 400))
    for p_, path in enumerate(splits):
        d = np.load(_repo_io(path)); di, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(di, fold) if f != k]; va = [i for i, f in zip(di, fold) if f == k]
            O[p_, [sid.index(i) for i in va]] = fp(kind, X[[pos[i] for i in tr]], np.array([lab[i] for i in tr]), X[[pos[i] for i in va]])
    return O
y, g, fail = N.y, N.g, N.fail; ng = ~g & ~fail[0]; base = np.load(_repo_io('dev-combo.npy'))
res = {}
for tag in ('none', 'phys', 'rand'):
    ex = None if tag == 'none' else EX[tag]
    res[tag] = dict(ridge=oof('u1dmavgH2Z', 'ridge', ex), logridge=np.exp(np.mean([np.log(oof(v, 'log', ex)) for v in V], 0)), gmm=oof('u1dmavgH2Z', 'gmm', ex, gmm=True)); print(tag, flush=True)
rb = lambda a: np.sqrt(((a - y) ** 2)[:, ng].mean())
for tag in ('phys', 'rand'):
    print(tag, ' '.join(f'{k} {rb(res["none"][k]):.3f}->{rb(res[tag][k]):.3f}' for k in res['none']))
    new = base + np.where(ng, sum(w * (res[tag][k] - res['none'][k]) for k, w in (('ridge', 0.128), ('logridge', 0.128), ('gmm', 0.15))), 0)
    report(f'  076 + {tag} cols in plain partners', new, base=base)
np.savez(_repo_io('h9-partners.npz'), **{f'{t}_{k}': v for t, d in res.items() for k, v in d.items()})
