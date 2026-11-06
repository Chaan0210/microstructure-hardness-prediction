# residual variant of push-1200/tabaug_dev.py: TabPFN fits y - col (cross-fitted DTA log-Ridge), prediction = col + TabPFN
"""TabPFN (044 slot features, one U-Net variant) with an extra column = sim-augmented log-Ridge prediction (degraded copies with measured noise >= 12
of clean/mid train images whose source is not in the relevant validation fold). Training rows get inner-5-fold OOF values (inner fits use the
inner-training real rows + allowed copies), validation rows the outer-training fit. Official dev partitions/folds/seeds (as dev900.py).
usage: tabaug_dev.py VARIANT SETS DEVICE  -> dev-<VARIANT>-anghcorTA.npz (push-300 dev format)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
P = Path(_repo_io('../..')); v, sets, device = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
SLOTF = sys.argv[4] if len(sys.argv) > 4 else '../push-800/feat-{v}-anghcor-train.npz'; OUTTAG = sys.argv[5] if len(sys.argv) > 5 else 'anghcorTA'   # slot feature file (column always from anghcor)
lab = {r.split(',')[0]: float(r.split(',')[1]) for r in open(_repo_path(P, 'data/train.csv')).read().split()[1:]}
F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
FS = np.load(_repo_io(SLOTF.format(v=v)), allow_pickle=True); posS = {str(i): k for k, i in enumerate(FS['ids'])}; XSA = FS['X'].astype(float)
SX, SY, SID = [], [], []
for s in sets:
    S = np.load(_repo_io(f'../push-1200/synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); X_ = S['X'].astype(float); m = X_[:, keys.index('noise')] * 255 >= 12
    SX.append(X_[m]); SY += list(np.array([lab[str(i)] for i in S['ids']])[m]); SID += list(np.array([str(i) for i in S['ids']])[m])
SX = np.vstack(SX); SY = np.array(SY); SID = np.array(SID)
LA = [0.01, 0.1, 0.3, 1, 2, 3, 10, 30, 100]
def auglin(Xtr, ytr, idtr, Xev, exclude):
    keep = np.array([s not in exclude for s in SID]); Xt = np.vstack([Xtr, SX[keep]]); yt = np.r_[ytr, SY[keep]]
    return np.exp(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=LA)).fit(Xt, np.log(yt)).predict(Xev))
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'),
          '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
o3 = np.load(_repo_io('../grain-cellpose-220/oof-v3.npz')); save = {}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; XS = XSA[[posS[i] for i in ids]]; a = f'B_v3feat_v3mask_{name}_'; p = 2 * o3[a + 'b'] - o3[a + 'g']
    g = np.zeros(len(y)); idsa = np.array(ids)
    for k in range(5):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]; vset = set(idsa[va]); col = np.zeros(len(y))
        col[va] = auglin(X[tr], y[tr], idsa[tr], X[va], vset)
        for itr, iva in KFold(5, shuffle=True, random_state=k).split(tr):
            col[tr[iva]] = auglin(X[tr[itr]], y[tr[itr]], idsa[tr[itr]], X[tr[iva]], vset | set(idsa[tr[iva]]))
        Xf = XS; ps = []
        for s in (0, 1, 2):
            m = harness.tabpfn_new('v3.5', 100 * s + k, device); m.fit(Xf[tr], y[tr] - col[tr]); ps.append(col[va] + m.predict(Xf[va]))
        g[va] = np.mean(ps, 0)
    failed = np.nan_to_num(X[:, keys.index('n_obj')], nan=0) < 20
    nz = X[:, keys.index('noise')] * 255
    print(name, 'grain %.4f noisy %.3f' % (np.sqrt(((g - y) ** 2).mean()), np.sqrt(((g - y)[nz >= 12] ** 2).mean())), flush=True)
    save.update({f'{name}_ids': idsa, f'{name}_y': y, f'{name}_fold': fold, f'{name}_p021': p, f'{name}_g': g, f'{name}_failed': failed})
np.savez(_repo_io(f'dev-{v}-{OUTTAG}.npz'), **save)
