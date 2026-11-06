"""Measurement quality of the geometry-CNN on REAL noisy train images (pixel noise >= 12; never seen by the CNN): correlation with hardness of
the CNN phase-area fraction vs the production phase features, and 10-fold CV R2 on noisy rows of Ridge on (a) the 13 CNN columns + noise,
(b) 6 production phase/size columns + noise, (c) both. usage: geom_check.py TAG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
tag = sys.argv[1]
G = np.load(_repo_io(f'geom-cnn{tag}-train.npz')); gid = [str(i) for i in G['ids']]; gk = [str(k) for k in G['keys']]; GX = G['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in gid])
P = np.load(_repo_io('/private/tmp/claude-501/-Users-dongchan-Desktop-PythonWorkspace-microstructure-2026/f62e347b-cb1a-4c17-8d54-2bf505be6706/scratchpad/presets.npz')); nzm = dict(zip(P['tr'], P['ntr'])); nz = np.array([nzm[i] for i in gid])
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; fp = {str(i): k for k, i in enumerate(F['ids'])}; FX = F['X'][[fp[i] for i in gid]].astype(float)
for nm, m in (('clean (in-sample for CNN)', nz < 9.5), ('mid', (nz >= 9.5) & (nz < 12)), ('noisy', nz >= 12)):
    out = [f'{c} {np.corrcoef(FX[m, fk.index(c)][np.isfinite(FX[m, fk.index(c)])], y[m][np.isfinite(FX[m, fk.index(c)])])[0, 1]:+.3f}' for c in ('u_ph', 'z_f25', 'sh_cor15')]
    print(f'{nm:26s} corr(y): cnn phase_area {np.corrcoef(GX[m, gk.index("phase_area_frac")], y[m])[0, 1]:+.3f}  cnn phase_count {np.corrcoef(GX[m, gk.index("phase_count_frac")], y[m])[0, 1]:+.3f} | ' + '  '.join(out))
m = nz >= 12; prod = ['u_ph', 'z_f25', 'sh_cor15', 'm_ph_hard', 'n_obj', 'aw_inv_sqrt_d', 'het4_u', 'orient_coherence', 'grain_logA_std', 'noise']
A = np.c_[GX[m], FX[m, fk.index('noise')]]; Bm = FX[m][:, [fk.index(c) for c in prod]]
def r2(Z):
    v = []
    for s in range(10):
        pr = cross_val_predict(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 13))), Z, y[m], cv=KFold(10, shuffle=True, random_state=s)); v.append(1 - np.mean((pr - y[m]) ** 2) / y[m].var())
    return np.mean(v)
print(f'noisy rows (n={m.sum()}) Ridge CV R2: CNN13+noise {r2(A):.3f} | production 10 cols {r2(Bm):.3f} | both {r2(np.c_[A, Bm]):.3f} | all 98 {r2(FX[m]):.3f} | 98+CNN13 {r2(np.c_[FX[m], GX[m]]):.3f}')
