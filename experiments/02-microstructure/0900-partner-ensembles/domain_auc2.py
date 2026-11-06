"""domain_auc.py variant: real-vs-sim AUC for parameter sets given as JSON {name: params}. usage: domain_auc2.py JSON
Generic image statistics NOT used to fit realdeg2: battery T (texture/frequency only: gradient, LoG, high-pass moments, radial PSD shares) and
battery A (T + darkness quantiles at sigma 1 and 2). 5-fold stratified logistic regression, 3 repeats; AUC 0.5 = indistinguishable.
Simulated sets: ours (realdeg2 DEFAULT), external (comparison only), old U-Net training branch (phase_unet4 default degrade), noise only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi, stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(H)); from realdeg2 import degrade2
from calib_own import surface, ids, Mc, nz
YY, XX = np.mgrid[0:256, 0:256] / 255.0 - 0.5; yy, xx = YY, XX
fr = np.fft.fftfreq(256); RAD = np.sqrt(fr[:, None] ** 2 + fr[None, :] ** 2); BANDS = [0, 0.03, 0.06, 0.1, 0.16, 0.25, 0.36, 0.71]

def feats(img):
    S = surface(ndi.gaussian_filter(img, 3.0)); x = img / S
    g = ndi.gaussian_gradient_magnitude(x, 1.0); T = list(np.quantile(g, [0.5, 0.9, 0.99]))
    for s in (1.0, 2.0, 3.0): T.append(np.abs(ndi.gaussian_laplace(x, s)).mean())
    hp = x - ndi.gaussian_filter(x, 2.0); T += [hp.std(), stats.kurtosis(hp.ravel()), stats.skew(hp.ravel())]
    F = np.abs(np.fft.fft2(x - x.mean())) ** 2; tot = F.sum(); T += [F[(RAD >= a) & (RAD < b)].sum() / tot for a, b in zip(BANDS[:-1], BANDS[1:])]
    A = list(T)
    for s in (1.0, 2.0): z = 1 - ndi.gaussian_filter(x, s); A += list(np.quantile(z, [0.01, 0.1, 0.5, 0.9, 0.99]))
    return np.array(T), np.array(A)

import os
prms = json.loads(sys.argv[1])   # {name: params}
clean = [n for n in range(len(ids)) if nz[n] < 9.5]; real = [n for n in range(len(ids)) if nz[n] >= 12]; pool = nz[nz >= 12]
RT, RA = zip(*[feats(np.array(Image.open(_repo_path(P, f'data/train/{ids[n]}.png'))).astype(np.float64)) for n in real])
for nm, prm in prms.items():
    r = np.random.RandomState(3); ST, SA = [], []
    for n in clean:
        a = np.array(Image.open(_repo_path(P, f'data/train/{ids[n]}.png'))).astype(np.float64); t, al = feats(degrade2(a, Mc['masks'][n].astype(int), r, prm, noise=float(r.choice(pool)))); ST.append(t); SA.append(al)
    out = {}
    for bn, Rr, Ss in (('T', RT, ST), ('A', RA, SA)):
        X = np.nan_to_num(np.vstack([np.array(Rr), np.array(Ss)])); y = np.r_[np.ones(len(Rr)), np.zeros(len(Ss))]
        out[bn] = round(float(np.mean([roc_auc_score(y, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=k), method='predict_proba')[:, 1]) for k in range(3)])), 3)
    print(nm, json.dumps(prm), out, flush=True)
