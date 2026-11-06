"""Segmentation-free phase measurement: background-corrected (Gaussian 24 subtracted) and Gaussian-smoothed (sigma s) intensities -> 3-component
1-D GMM (pore / dark phase / matrix) -> dark-phase area fraction, pore fraction, phase contrast; also Otsu-split fraction. Label-free, per image.
usage: sfphase.py train|test -> sfphase-<split>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture
P = Path(_repo_io('../..')); sp = sys.argv[1]
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv' if sp == 'train' else 'data/sample_submission.csv')))]
out = []
for i in ids:
    I = np.asarray(Image.open(_repo_path(P, f'data/{sp}/{i}.png')).convert('L'), np.float64); B = ndi.gaussian_filter(I, 24); R = I - B + B.mean()
    row = []
    for s in (1.5, 3.0):
        S = ndi.gaussian_filter(R, s)[8:-8, 8:-8].ravel(); x = S[::3].reshape(-1, 1)
        gm = GaussianMixture(3, covariance_type='spherical', random_state=0, n_init=2).fit(x); o = np.argsort(gm.means_.ravel()); mu = gm.means_.ravel()[o]; w = gm.weights_[o]; sd = np.sqrt(gm.covariances_[o])
        lab = np.argsort(o)[gm.predict(S.reshape(-1, 1))]   # 0 darkest
        g2 = GaussianMixture(2, covariance_type='spherical', random_state=0, n_init=2).fit(x); o2 = np.argsort(g2.means_.ravel())
        row += [w[0], w[1], (lab == 1).mean(), (lab == 0).mean(), mu[2] - mu[1], mu[2] - mu[0], sd[1], sd[2], g2.weights_[o2][0], np.diff(g2.means_.ravel()[o2])[0]]
    out.append(row)
keys = [f'{n}_s{s}' for s in (1.5, 3.0) for n in ('w_pore', 'w_dark', 'f_dark', 'f_pore', 'c_dark', 'c_pore', 'sd_dark', 'sd_mat', 'w2_dark', 'c2')]
np.savez(_repo_io(f'sfphase-{sp}.npz'), ids=np.array(ids), X=np.array(out), keys=np.array(keys)); print('done', sp)
