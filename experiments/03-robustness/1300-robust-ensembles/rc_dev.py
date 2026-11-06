"""Regression calibration (errors-in-variables) for noisy rows, TabPFN v3.5 member on the Study900 dev folds.
Per outer fold: multi-output RidgeCV map x_clean ~ x_deg learned on degraded copies (deg..deg4, measured noise >= 12) whose SOURCE image is in the
training fold (pairs: copy features -> the source's own clean features); every real row with noise >= 12 (train and validation) gets
x_hat = E[x_clean | x_noisy] for all columns except 'noise' (kept, so the model still knows the row is noisy); TabPFN is fit on the training rows
(clean rows unchanged, noisy rows mapped) and predicts the validation rows. Seeds 0,1,2 (random_state 100*sd + fold).
usage: rc_dev.py VARIANT(u1dmavg) TAG [MODE: full|nb]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, json, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300')); import harness
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
v, tag = sys.argv[1], sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else 'full'
P = Path(_repo_io('../..')); lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sid = load(3)['ids']; F = np.load(_repo_io(f'../push-800/feat-{v}H2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float); nzr = np.nan_to_num(XA[:, ni] * 255, nan=0)
SX, SID = [], []
for s in ['deg', 'deg2', 'deg3', 'deg4']:
    S = np.load(_repo_io(f'../push-1200/synthrows-{s}-{v}H2Z-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
SX = np.vstack(SX); SID = np.array(SID)
cols_in = [j for j in range(len(keys))]; cols_out = [j for j in range(len(keys)) if j != ni]
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
O = np.zeros((3, 400)); noisy_all = nzr >= 12
for p_, path in enumerate(splits):
    d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
    for k in range(5):
        tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; trs = set(tr)
        kp = np.array([s in trs for s in SID]); Xd = SX[kp]; Xc = XA[[pos[s] for s in SID[kp]]]
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(np.vstack([Xd, XA[[pos[i] for i in tr]]]))
        sx = StandardScaler().fit(imp.transform(Xd)); sy = StandardScaler().fit(imp.transform(Xc)[:, cols_out])
        mp = RidgeCV(alphas=np.logspace(-2, 3, 11)).fit(sx.transform(imp.transform(Xd)), sy.transform(imp.transform(Xc)[:, cols_out]))
        def mapped(rows):
            X = imp.transform(XA[rows]).copy(); nzm = noisy_all[rows]
            if nzm.any(): X[np.ix_(nzm, cols_out)] = sy.inverse_transform(mp.predict(sx.transform(X[nzm])))
            return X
        jt = np.array([pos[i] for i in tr]); jv = np.array([pos[i] for i in va]); Xt, Xv = mapped(jt), mapped(jv); yt = np.array([lab[i] for i in tr]); ps = []
        for sd in (0, 1, 2):
            m = harness.tabpfn_new('v3.5', 100 * sd + k, 'mps'); m.fit(Xt, yt); ps.append(m.predict(Xv))
        O[p_, [sid.index(i) for i in va]] = np.mean(ps, 0)
    print('partition', p_, 'done', flush=True)
np.savez(_repo_io(f'rc-{tag}.npz'), oof=O)
