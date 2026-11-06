"""Final (500 train rows) Ridge / log-Ridge partners with label-preserving degraded copies of clean train images added to the fit
(push-400/synth/train_<set>, rows built by synth_anghcor.py), predicting the 1000 test rows. Exactly the production settings:
RidgeCV(alphas [0.1,1,10,100,1000]) on feat-u1dmavgH2Z-anghcor (median imputation keep_empty, standardisation) and
exp(log-target Ridge alpha 2) geometric-mean over the 4 variant sets. With SETS=none the production test predictions must be reproduced.
usage: final_aug.py SETS(comma|none) OUT"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]
sets = [] if sys.argv[1] == 'none' else sys.argv[1].split(','); out = sys.argv[2]
NMIN = float(__import__('os').environ.get('NOISEMIN', '-1'))
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P0, 'data/train.csv')))}
order = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data/sample_submission.csv')))]
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
def data(v):
    Tr = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); Te = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-test.npz'), allow_pickle=True)
    assert [str(i) for i in Te['ids']] == order and list(Tr['keys']) == list(Te['keys'])
    X = Tr['X'].astype(float); y = np.array([lab[str(i)] for i in Tr['ids']]); X0 = X; pos = {str(i): j for j, i in enumerate(Tr['ids'])}; ni = list(Tr['keys']).index('noise')
    CX, CY, CS = [], [], []
    for s in sets:
        S = np.load(_repo_path(H, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); assert list(S['keys']) == list(Tr['keys'])
        SX = S['X'].astype(float); m = SX[:, ni] * 255 >= NMIN   # NOISEMIN env: keep only copies with measured noise >= NMIN
        CX.append(SX[m]); CY += list(np.array([lab[str(i)] for i in S['ids']])[m]); CS += [pos[str(i)] for i in np.array(S['ids'])[m]]
    if CX:
        CX = np.vstack(CX); LAM = float(__import__('os').environ.get('SHIFTFIX', '0'))   # SHIFTFIX=lam: copies += lam*(real noisy-clean shift - paired copy-source shift), 500 train rows only
        if LAM > 0:
            nz0 = np.nan_to_num(X0[:, ni] * 255, nan=0); real = np.nanmean(X0[nz0 >= 12], 0) - np.nanmean(X0[nz0 < 9.5], 0); sim = np.nanmean(CX - X0[CS], 0)
            fix = np.nan_to_num(real - sim); fix[ni] = 0; CX = CX + LAM * fix
        X = np.vstack([X, CX]); y = np.r_[y, CY]
    return X, y, Te['X'].astype(float), [str(k) for k in Te['keys']]
X, y, Xte, keys = data('u1dmavgH2Z')
ALPHAS = [float(a) for a in __import__('os').environ.get('ALPHAS', '0.1,1,10,100,1000').split(',')]; LA = __import__('os').environ.get('LOGALPHAS')
rm = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X, y)
ridge = rm.predict(Xte); logs = []
for v in V:
    X, y, Xte_v, _ = data(v); logs.append(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=[float(a) for a in LA.split(',')]) if LA else Ridge(alpha=2.0)).fit(X, np.log(y)).predict(Xte_v))
logridge = np.exp(np.mean(logs, 0)); noise = Xte[:, keys.index('noise')] * 255
np.savez(_repo_path(H, f'{out}.npz'), ids=np.array(order), ridge=ridge, logridge=logridge, noise=noise, alpha=rm[-1].alpha_, n_fit=len(y))
print(out, 'alpha', rm[-1].alpha_, 'n_fit rows', len(y), 'test noisy>=12', int((np.nan_to_num(noise) >= 12).sum()))
