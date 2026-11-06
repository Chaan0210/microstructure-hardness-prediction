"""Group-penalised plain RidgeCV (u1dmavgH2Z) for non-gated rows inside 073: the 13 het+intensity columns are multiplied by s after standardisation
(penalty on their coefficients x 1/s^2). Pre-registered arms: s = 1 (full), s = 1/sqrt(10) (penalty x10), s = 0 (delete = 073). dev + holdout."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import RidgeCV
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
import nsgain as N
from robust_eval import report
P = Path(_repo_io('../..')); sid = load(3)['ids']; dev = set(sid); lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
G = json.load(open(_repo_io('abl/groups.json'))); drop = set(G['intensity'].split() + G['het'].split())
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): j for j, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
gm = np.array([k in drop for k in keys])
def mk(s):
    sc = np.where(gm, s, 1.0); return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), FunctionTransformer(lambda X: X * sc), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
y, g, C, fail, nz = N.y, N.g, N.C, N.fail, N.nz; ng = ~g & ~fail[0]; c73 = np.load(_repo_io('dev-nb-candidate.npy')); Z = np.load(_repo_io('nb-partners.npz'))
r73 = Z['ridge_nb_dev']   # the 073 plain ridge (s = 0 equivalent: columns removed)
res = {}
for s in (1.0, 10 ** -0.5, 0.0):
    O = np.zeros((3, 400))
    for p_, path in enumerate(splits):
        d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]
            O[p_, [sid.index(i) for i in va]] = mk(s).fit(XA[[pos[i] for i in tr]], [lab[i] for i in tr]).predict(XA[[pos[i] for i in va]])
    res[s] = O; new = np.where(ng, c73 + 0.128 * (O - r73), c73)
    print(f's {s:.3f}: plain ridge non-gated RMSE {np.sqrt(((O - y) ** 2)[:, ng].mean()):.3f} (073 nb {np.sqrt(((r73 - y) ** 2)[:, ng].mean()):.3f}; max|O(s=0) - nb| {np.abs(res[s] - r73).max() if s == 0 else float("nan"):.2e})')
    report(f'   s {s:.3f} vs 073', new, base=c73)
# holdout: fit on dev-400, predict holdout non-gated rows (ridge alone)
hd = [i for i in pos if i not in dev]; td = [i for i in pos if i in dev]; yh = np.array([lab[i] for i in hd]); ngh = np.nan_to_num(XA[[pos[i] for i in hd], keys.index('noise')] * 255, nan=0) < 12
ph = {s: mk(s).fit(XA[[pos[i] for i in td]], [lab[i] for i in td]).predict(XA[[pos[i] for i in hd]]) for s in res}
for s in res:
    dd = (ph[s] - yh)[ngh] ** 2 - (ph[0.0] - yh)[ngh] ** 2; print(f'holdout non-gated (n={ngh.sum()}) s {s:.3f}: ridge RMSE {np.sqrt(((ph[s] - yh)[ngh] ** 2).mean()):.3f} | dMSE vs delete {dd.mean():+.2f} ± {dd.std() / np.sqrt(ngh.sum()):.2f}')
