"""Clean rows only (249 train images, pixel noise < 9.5): do exact global geometry parameters (push-1000 geom-exact, from zero-shot masks) add to the
98 production columns? TabPFN-3.5 trained and evaluated on clean rows only, 5-fold x 3 repeats, identical folds for every feature set."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sys.path.insert(0, str(_repo_path(E, 'push-300'))); import harness
from sklearn.model_selection import KFold
G = np.load(_repo_path(E, 'push-1000/geom-exact-train.npz'), allow_pickle=True); ids = [str(i) for i in G['ids']]; XG = G['X'].astype(float)
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(E.parent, 'data/train.csv')))}; y = np.array([lab[i] for i in ids])
F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); K = [str(k) for k in F['keys']]; pos = {str(i): j for j, i in enumerate(F['ids'])}
X98 = F['X'][[pos[i] for i in ids]].astype(float)
extra = ['het4_u', 'het4_u_mx', 'ang_c4', 'ang_s4', 'm_el_pix', 'sizefield_std_s32', 'sh_cor15', 'noise', 'u_ph', 'z_f25']
sets = {'A98': X98, 'B13+extra': np.c_[XG, X98[:, [K.index(c) for c in extra]]], 'C98+13': np.c_[X98, XG]}
P = {k: np.zeros((3, len(y))) for k in sets}
for r in range(3):
    for f, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=100 + r).split(y)):
        for k, X in sets.items():
            m = harness.tabpfn_new('v3.5', 10 * r + f, 'mps'); m.fit(X[tr], y[tr]); P[k][r, va] = m.predict(X[va])
    print('repeat', r, {k: round(float(np.sqrt(((P[k][r] - y) ** 2).mean())), 3) for k in sets}, flush=True)
for k in sets: print(k, 'RMSE', np.sqrt(((P[k] - y) ** 2).mean()).round(4), ' per repeat', np.sqrt(((P[k] - y) ** 2).mean(1)).round(3))
d = ((P['A98'] - y) ** 2).mean(0) - ((P['C98+13'] - y) ** 2).mean(0); bs = [d[np.random.RandomState(s).randint(0, len(d), len(d))].mean() for s in range(2000)]
print('C vs A dMSE', d.mean().round(3), 'boot 95%', np.percentile(bs, [2.5, 97.5]).round(3))
np.savez(_repo_path(H, 'clean_exact.npz'), ids=np.array(ids), y=y, **P)
