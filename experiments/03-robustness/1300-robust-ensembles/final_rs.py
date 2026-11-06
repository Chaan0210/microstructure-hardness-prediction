# residual variant of push-1200/final_ta.py: TabPFN on (slot features, y - cross-fitted DTA log-Ridge), test = DTA log-Ridge + TabPFN
"""Final 044-slot TabPFN with the sim-augmented log-Ridge column (as tabaug_dev.py): train column = 5-fold OOF over the 500 train rows (each inner
fit uses its training rows + degraded copies with measured noise >= 12 whose source is not in that inner validation fold); test column = fit on all
500 rows + all such copies. TabPFN-3.5, seeds 0/100/200 (finalize_multi.py convention), mean over the 4 U-Net variant sets.
usage: final_ta.py SETS DEVICE -> fin-ta.npz (ids, g44ta [1000])"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
P = Path(_repo_io('../..')); sets, device = sys.argv[1].split(','), sys.argv[2]
SLOT = sys.argv[3] if len(sys.argv) > 3 else '../push-800/feat-{v}-anghcor-{split}.npz'; OUT = sys.argv[4] if len(sys.argv) > 4 else 'fin-ta'   # slot features (column always from anghcor)
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
order = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/sample_submission.csv')))]
LA = [0.01, 0.1, 0.3, 1, 2, 3, 10, 30, 100]; preds = []
for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']:
    Tr = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); Te = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-test.npz'), allow_pickle=True)
    keys = [str(k) for k in Tr['keys']]; assert [str(i) for i in Te['ids']] == order
    X = Tr['X'].astype(float); ids = np.array([str(i) for i in Tr['ids']]); y = np.array([lab[i] for i in ids]); Xe = Te['X'].astype(float)
    STr = np.load(_repo_io(SLOT.format(v=v, split='train')), allow_pickle=True); STe = np.load(_repo_io(SLOT.format(v=v, split='test')), allow_pickle=True)
    assert [str(i) for i in STr['ids']] == list(ids) and [str(i) for i in STe['ids']] == order; XS, XSe = STr['X'].astype(float), STe['X'].astype(float)
    SX, SY, SID = [], [], []
    for s in sets:
        S = np.load(_repo_io(f'../push-1200/synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); X_ = S['X'].astype(float); m = X_[:, keys.index('noise')] * 255 >= 12
        SX.append(X_[m]); SY += list(np.array([lab[str(i)] for i in S['ids']])[m]); SID += list(np.array([str(i) for i in S['ids']])[m])
    SX = np.vstack(SX); SY = np.array(SY); SID = np.array(SID)
    def auglin(Xtr, ytr, Xev, exclude):
        keep = np.array([s not in exclude for s in SID]); Xt = np.vstack([Xtr, SX[keep]]); yt = np.r_[ytr, SY[keep]]
        return np.exp(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=LA)).fit(Xt, np.log(yt)).predict(Xev))
    col = np.zeros(len(y))
    for itr, iva in KFold(5, shuffle=True, random_state=0).split(X): col[iva] = auglin(X[itr], y[itr], X[iva], set(ids[iva]))
    cte = auglin(X, y, Xe, set())
    ps = []
    for sd in (0, 1, 2):
        m = harness.tabpfn_new('v3.5', 100 * sd, device); m.fit(XS, y - col); ps.append(cte + m.predict(XSe))
    preds.append(np.mean(ps, 0)); print(v, 'done', flush=True)
np.savez(_repo_io(f'{OUT}.npz'), ids=np.array(order), g44ta=np.mean(preds, 0), gta=np.mean(preds, 0), per_variant=np.array(preds)); print(f'saved {OUT}.npz')
