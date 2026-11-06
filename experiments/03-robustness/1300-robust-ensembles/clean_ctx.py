"""Context ablation for the linear partners: the plain partners (used on non-gated rows) trained on (a) all real rows (061) vs (b) only real rows with
noise < 12 (no noisy-measurement rows -> no errors-in-variables attenuation of the clean-domain coefficients) vs (c) < 12 rows + noisy rows down-weighted 0.3.
RidgeCV u1dmavgH2Z + logRidge alpha 2 x 4 variants; DTA partners unchanged. Study900 dev folds."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
P = Path(_repo_io('../..')); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; MODE = sys.argv[1]; CUT = float(sys.argv[2]) if len(sys.argv) > 2 else 12
def mk(kind): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
def oof(v, kind):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
    pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float); nzr = np.nan_to_num(XA[:, ni] * 255, nan=0); O = np.zeros((3, 400))
    for p_, path in enumerate(splits.values()):
        d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]
            jt = np.array([pos[i] for i in tr]); w = np.ones(len(jt)); low = nzr[jt] < CUT
            if MODE == 'clean': jt, w = jt[low], w[low]
            elif MODE == 'down': w[~low] = 0.3
            yy = np.array([lab[i] for i in np.array(tr)[np.isin(np.array([pos[i] for i in tr]), jt)]]) if MODE == 'clean' else np.array([lab[i] for i in tr])
            m = mk(kind); m.fit(XA[jt], np.log(yy) if kind == 'log' else yy, **{m.steps[-1][0] + '__sample_weight': w}); pr = m.predict(XA[[pos[i] for i in va]])
            O[p_, [sid.index(i) for i in va]] = np.exp(pr) if kind == 'log' else pr
    return O
Z = np.load(_repo_io('rr-none.npz'))
np.savez(_repo_io(f'rr-ctx{MODE}{int(CUT)}.npz'), ridge_0=oof('u1dmavgH2Z', 'ridge'), logridge_0=np.exp(np.mean([np.log(oof(v, 'log')) for v in V], 0)), ridge_1=Z['ridge_1'], logridge_1=Z['logridge_1'])
print('saved', MODE, CUT)
