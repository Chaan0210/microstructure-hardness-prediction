"""Paired residual transfer (PRT), dev. Outer TabPFN f (outer-training real rows, u1dmavgH2Z-anghcor) predicts degraded copies (sources outside the
validation fold) and validation rows. Label-free degradation shift per copy: D = f(copy) - t(source), t = teacher cache (inner cross-fitted TabPFN
on the clean source features, tt_dev.py). Linear map g: D ~ copy features (Ridge, CV alpha); output per validation row: f, g.
usage: prt_dev.py SETS TAG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, csv, warnings, numpy as np
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, _repo_io('../push-300')); import harness
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); v = 'u1dmavgH2Z'; sets, tag = sys.argv[1].split(','), sys.argv[2]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
ids_all = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(ids_all)}; XA = F['X'].astype(float); yA = np.array([lab[i] for i in ids_all])
SX, SID = [], []
for s in sets:
    S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
SX = np.vstack(SX); SID = np.array(SID)
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sid = load(3)['ids']; TC = dict(np.load(_repo_io(f'teacher-{v}.npz')))
out = {k: np.zeros((3, 400)) for k in ('f', 'g', 'gconst')}
for p_, (name, path) in enumerate(splits.items()):
    d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
    for k in range(5):
        tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va); keep = np.array([s not in vs for s in SID])
        tmap = dict(zip([str(i) for i in TC[f'{name}_{k}_ids']], TC[f'{name}_{k}_t']))
        m = harness.tabpfn_new('v3.5', 0, 'mps').fit(XA[[pos[i] for i in tr]], yA[[pos[i] for i in tr]])
        Xc = SX[keep]; pr = m.predict(np.vstack([Xc, XA[[pos[i] for i in va]]])); fc, fv = pr[:len(Xc)], pr[len(Xc):]
        D = fc - np.array([tmap[s] for s in SID[keep]])
        gm = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[1, 10, 100, 1000, 1e4])).fit(Xc, D)
        jj = [sid.index(i) for i in va]; out['f'][p_, jj] = fv; out['g'][p_, jj] = gm.predict(XA[[pos[i] for i in va]]); out['gconst'][p_, jj] = D.mean()
        print(name, k, 'mean shift %.2f sd %.2f, alpha %g' % (D.mean(), D.std(), gm[-1].alpha_), flush=True)
np.savez(_repo_io(f'prt-{tag}.npz'), **out)
