"""Dev (fixed rules, no fitted weights) for submission options on top of 039/044:
 (i) perturbation ensemble = mean grain OOF over the 7 random-extra-column runs (variance reduction, not selection)
 (ii) Ridge 20%: grain = 0.8*TabPFN + 0.2*RidgeCV (median-impute, standardize, alphas logspace(-2,4,40); same as push-700/ml_compare.py) on the same columns
 (iii) W grid for reference. Pooled over 3 partitions, 4-variant mean."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, json
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
E = str(_repo_path(_REPO_ROOT, 'experiments')) + '/'
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; parts = ['original', '20261031', '20261101']
def load(tag): return [np.load(_repo_io(E + (f'push-600/dev-{v}-angh.npz' if tag == 'base' else f'push-800/dev-{v}-angh{tag}.npz')), allow_pickle=True) for v in V]
D0 = load('base'); ref = [str(i) for i in D0[0]['original_ids']]
def grain(tag, n): return np.mean([d[f'{n}_g'] for d in load(tag)], 0)
def ridge_oof(featfile, n):
    F = np.load(_repo_io(featfile), allow_pickle=True); pos = {str(i): k for k, i in enumerate(F['ids'])}
    ids = [str(i) for i in D0[0][f'{n}_ids']]; X = F['X'][[pos[i] for i in ids]].astype(float); y = D0[0][f'{n}_y']; fold = D0[0][f'{n}_fold']; o = np.zeros(len(y))
    for k in range(5):
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 40))).fit(X[fold != k], y[fold != k]); o[fold == k] = m.predict(X[fold == k])
    return o
def score(gfun, W=0.75):
    se = []; per = []
    for n in parts:
        ids = [str(i) for i in D0[0][f'{n}_ids']]; ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]
        y, p, f = D0[0][f'{n}_y'], D0[0][f'{n}_p021'], D0[0][f'{n}_failed']; g = gfun(n)
        e = ((1 - W) * p + W * np.where(f, p, g) - y) ** 2; se.append(e[j]); per.append(np.sqrt(e.mean()))
    return float(np.sqrt(np.mean(se))), per
rnds = ['rnd'] + [f'rnd{k}' for k in range(1, 7)]
R39 = {n: ridge_oof(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz', n) for n in parts}
R44 = {n: np.mean([ridge_oof(E + f'push-800/feat-{v}-anghcor-train.npz', n) for v in V], 0) for n in parts}
R39 = {n: np.mean([ridge_oof(E + f'push-700/feat-{v}-angh-train.npz', n) for v in V], 0) for n in parts}
opts = {
 '039': lambda n: grain('base', n),
 '039 + perturbation ens (7 rnd runs)': lambda n: np.mean([grain(t, n) for t in rnds], 0),
 '039 + ridge20': lambda n: 0.8 * grain('base', n) + 0.2 * R39[n],
 '044': lambda n: grain('cor', n),
 '044 + ridge20': lambda n: 0.8 * grain('cor', n) + 0.2 * R44[n],
 'ridge only (044 cols)': lambda n: R44[n],
}
for k, f in opts.items():
    r = {W: score(f, W) for W in (0.75, 0.8, 0.85, 0.9, 1.0)}
    print(f'{k:38s} ' + '  '.join(f'W{W}: {r[W][0]:.4f}' for W in r) + '   | W0.75 per-part ' + ' '.join(f'{v:.3f}' for v in r[0.75][1]))
print('individual rnd runs W0.75:', ' '.join(f'{score(lambda n, t=t: grain(t, n))[0]:.4f}' for t in rnds))
