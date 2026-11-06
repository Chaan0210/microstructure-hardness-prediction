"""Feature-group ablation of the linear partners (RidgeCV u1dmavgH2Z and logRidge alpha 2 u1dmavgH2Z; plain on real rows, DTA with d1234 copies
noise >= 12), Study900 dev folds. Partner RMSE per band and the 061 blend effect when both partners are rebuilt without the group
(DTA on noise >= 12 rows, plain elsewhere, as in 061; logRidge here single-variant, so the blend uses the ratio new/full applied to 061's partner)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
import sys; sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
import nsgain as N
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
G = json.load(open(_repo_io('abl/groups.json'))); v = 'u1dmavgH2Z'
F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float); SX, SID = [], []
for s in ['deg', 'deg2', 'deg3', 'deg4']:
    S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
SX = np.vstack(SX); SID = np.array(SID)
def mk(kind): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
def oof(cols, kind, dta):
    O = np.zeros((3, 400))
    for p_, path in enumerate(splits.values()):
        d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va)
            X = XA[[pos[i] for i in tr]][:, cols]; yy = np.array([lab[i] for i in tr])
            if dta: kp = np.array([s not in vs for s in SID]); X = np.vstack([X, SX[kp][:, cols]]); yy = np.r_[yy, [lab[s] for s in SID[kp]]]
            m = mk(kind).fit(X, np.log(yy) if kind == 'log' else yy); pr = m.predict(XA[[pos[i] for i in va]][:, cols])
            O[p_, [sid.index(i) for i in va]] = np.exp(pr) if kind == 'log' else pr
    return O
y, g, c61, C, W, fail = N.y, N.g, N.c61, N.C, N.W, N.fail; f = fail[0]; nz = N.nz
cl = (nz < 11) & ~f
def part(cols):
    r = np.where(g, oof(cols, 'ridge', True), oof(cols, 'ridge', False)); l = np.where(g, oof(cols, 'log', True), oof(cols, 'log', False)); return r, l
allc = list(range(len(keys))); r0, l0 = part(allc)
rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
print(f'full: ridge clean {rb(r0, cl):.3f} noisy {rb(r0, g):.3f} | logr clean {rb(l0, cl):.3f} noisy {rb(l0, g):.3f}')
res = {}
for gname, cs in G.items():
    cols = [j for j, k in enumerate(keys) if k not in cs.split()]; r, l = part(cols)
    Cx = dict(C); Cx['ridge'] = C['ridge'] + (r - r0); Cx['logr'] = C['logr'] * (l / l0); new = N.blend(Cx)
    d = (new - y) ** 2 - (c61 - y) ** 2; pp = np.sqrt(((c61 - y) ** 2).mean(1)) - np.sqrt(((new - y) ** 2).mean(1))
    print(f'drop {gname:9s}: ridge clean {rb(r, cl):.3f} noisy {rb(r, g):.3f} | logr clean {rb(l, cl):.3f} noisy {rb(l, g):.3f} | blend gain {N.rm(c61) - N.rm(new):+.4f} '
          f'({"/".join(f"{x:+.3f}" for x in pp)}) clean dMSE {d[:, cl].mean():+.2f} noisy {d[:, g].mean():+.2f}', flush=True)
