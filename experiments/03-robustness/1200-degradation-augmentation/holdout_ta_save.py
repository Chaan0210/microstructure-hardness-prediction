"""(SAVE variant: also writes holdout-ta.npz with per-slot plain/TA holdout predictions)
Independent check of the TabPFN + sim-augmented-column idea on the 100 train rows outside the Study900 dev-400: fit on the 400 dev rows
(column for them = 5-fold OOF, copies whose source is a holdout row excluded everywhere), predict the 100 holdout rows; compare with the same
TabPFN without the column, per slot (038 H2Z, 039 angh, 044 anghcor), mean over the 4 U-Net variants and 3 seeds."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
dev = set(str(i) for i in np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['ids'])
sets = ['deg', 'deg2', 'deg3', 'deg4', 'deg5', 'deg6', 'deg7']; LA = [0.01, 0.1, 0.3, 1, 2, 3, 10, 30, 100]
SLOTS = {'038': '../push-500/feat-{v}-train.npz', '039': '../push-700/feat-{v}-angh-train.npz', '044': '../push-800/feat-{v}-anghcor-train.npz'}
res = {}
for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']:
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ids = np.array([str(i) for i in F['ids']])
    X = F['X'].astype(float); y = np.array([lab[i] for i in ids]); tr = np.array([i in dev for i in ids]); ho = ~tr
    SX, SY, SID = [], [], []
    for s in sets:
        S = np.load(_repo_io(f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); X_ = S['X'].astype(float); m = (X_[:, keys.index('noise')] * 255 >= 12) & np.array([str(i) in dev for i in S['ids']])
        SX.append(X_[m]); SY += list(np.array([lab[str(i)] for i in S['ids']])[m]); SID += list(np.array([str(i) for i in S['ids']])[m])
    SX = np.vstack(SX); SY = np.array(SY); SID = np.array(SID)
    def auglin(Xa, ya, Xe, excl):
        k = np.array([s not in excl for s in SID]); return np.exp(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=LA)).fit(np.vstack([Xa, SX[k]]), np.log(np.r_[ya, SY[k]])).predict(Xe))
    ti = np.where(tr)[0]; col = np.zeros(len(y))
    for a, b in KFold(5, shuffle=True, random_state=0).split(ti): col[ti[b]] = auglin(X[ti[a]], y[ti[a]], X[ti[b]], set(ids[ti[b]]))
    col[ho] = auglin(X[tr], y[tr], X[ho], set())
    nz = X[ho, keys.index('noise')] * 255; noisy = np.nan_to_num(nz) >= 12
    for sl, pat in SLOTS.items():
        FS = np.load(_repo_io(pat.format(v=v)), allow_pickle=True); assert [str(i) for i in FS['ids']] == list(ids); XS = FS['X'].astype(float)
        for tag, Xf in (('plain', XS), ('TA', np.c_[XS, col])):
            ps = []
            for sd in (0, 1, 2):
                m = harness.tabpfn_new('v3.5', 100 * sd, 'mps'); m.fit(Xf[tr], y[tr]); ps.append(m.predict(Xf[ho]))
            res.setdefault((sl, tag), []).append(np.mean(ps, 0))
    print(v, 'done', flush=True)
yh = y[ho]
for sl in SLOTS:
    for tag in ('plain', 'TA'):
        p = np.mean(res[(sl, tag)], 0); print(f'slot {sl} {tag:5s} holdout all {np.sqrt(((p - yh) ** 2).mean()):.3f}  noisy(n={noisy.sum()}) {np.sqrt(((p - yh)[noisy] ** 2).mean()):.3f}  clean+mid {np.sqrt(((p - yh)[~noisy] ** 2).mean()):.3f}')

np.savez(_repo_io('holdout-ta.npz'), ids=ids[ho], y=yh, noise=nz, **{f'{sl}_{tag}': np.mean(res[(sl, tag)], 0) for sl in SLOTS for tag in ('plain', 'TA')})
print('saved holdout-ta.npz')
