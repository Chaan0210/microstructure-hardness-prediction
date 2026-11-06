"""Branch 2 on dev (Study900 ids/partitions): do unsummarised 2-D correlation maps add information beyond the 98 summary columns?
Same learner everywhere (median impute + standardise + RidgeCV alphas 0.1..1000, everything fitted on outer-training rows only):
  A  X98                                     (control: the production Ridge partner input)
  B  X98 + radial means r=1..16 of the 4 maps + 4 amplitudes      (summary control)
  C  X98 + PCA-32 of the full maps (PCA fitted in fold) + amplitudes
  C2 X98 + all map values (4 x 33 x 33) + amplitudes
Reports standalone RMSE and the 053 structure with its RidgeCV partner swapped for each variant (A reproduces the swap baseline).
usage: acf_dev.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, _repo_io('../push-900'))
from blend_dev import load, rmse, parts
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
B = load(9); ids, y, FO, fail, p = B['ids'], B['y'], B['fold'], B['fail'], B['p021']
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); fp = {str(i): k for k, i in enumerate(F['ids'])}; X98 = F['X'][[fp[i] for i in ids]].astype(float)
Q = np.load(_repo_io('acf-train.npz')); qp = {str(i): k for k, i in enumerate(Q['ids'])}; A = Q['A'][[qp[i] for i in ids]].astype(float); amp = Q['amp'][[qp[i] for i in ids]].astype(float)
yy, xx = np.mgrid[-16:17, -16:17]; rr = np.rint(np.hypot(yy, xx)).astype(int)
RAD = np.stack([np.stack([A[:, c][:, rr == r].mean(1) for r in range(1, 17)], 1) for c in range(4)], 1).reshape(len(ids), -1)
FULL = A.reshape(len(ids), -1)
def ridge(): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 1e4]))
OUT = {}
for nm in ('A', 'B', 'C', 'C2'):
    P = np.zeros((3, 400))
    for k in range(3):
        for f in range(5):
            tr, va = FO[k] != f, FO[k] == f
            if nm == 'A': Z = X98
            elif nm == 'B': Z = np.c_[X98, RAD, amp]
            elif nm == 'C':
                pca = make_pipeline(StandardScaler(), PCA(32, random_state=0)).fit(FULL[tr]); Z = np.c_[X98, pca.transform(FULL), amp]
            else: Z = np.c_[X98, FULL, amp]
            P[k, va] = ridge().fit(Z[tr], y[tr]).predict(Z[va])
    OUT[nm] = P; np.savez(_repo_io('acf-dev.npz'), **OUT)
W = lambda a: np.where(fail, p, a); S3 = B['sets'][:3].mean(0)
def c53(r): return W(0.80 * (0.2 * p + 0.48 * B['g'] + 0.16 * r + 0.16 * B['logridge']) + 0.05 * S3 + 0.15 * B['gmm'])
base = c53(OUT['A'])
print('stored 053', round(rmse(W(0.80 * B['c051'] + 0.05 * S3 + 0.15 * B['gmm']), y), 4), ' 053 with recomputed RidgeCV (A)', round(rmse(base, y), 4))
for nm, P in OUT.items():
    c = c53(P); print(f'{nm:3s} alone {rmse(W(P), y):.3f}  053-swap {rmse(c, y):.4f} ({rmse(base, y) - rmse(c, y):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in parts(base, y) - parts(c, y)) + ')')
