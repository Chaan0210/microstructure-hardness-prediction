"""u1 vs u1b on real train images: band means of u_ph (phase over-detection on noisy rows), correlation with hardness per band, u1-u1b agreement."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
U1 = np.load(_repo_io('../push-400/unet-u1-train.npz'), allow_pickle=True); UB = np.load(_repo_io('../push-400/unet-u1b-train.npz'), allow_pickle=True)
k = [str(x) for x in U1['keys']]; assert [str(i) for i in U1['ids']] == [str(i) for i in UB['ids']]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[str(i)] for i in U1['ids']])
nz = np.load(_repo_io('../push-300/noise_train.npy')) * 255
for c in ('u_ph', 'u_ph_hard', 'u_ph_f32'):
    a, b = U1['X'][:, k.index(c)].astype(float), UB['X'][:, k.index(c)].astype(float)
    for nm, m in [('clean', nz < 9.5), ('mid', (nz >= 9.5) & (nz < 12)), ('noisy', nz >= 12)]:
        print(f'{c:10s} {nm:5s} mean u1 {a[m].mean():.3f} u1b {b[m].mean():.3f} | corr(y) u1 {np.corrcoef(a[m], y[m])[0, 1]:+.3f} u1b {np.corrcoef(b[m], y[m])[0, 1]:+.3f} | corr(u1,u1b) {np.corrcoef(a[m], b[m])[0, 1]:.3f}')
