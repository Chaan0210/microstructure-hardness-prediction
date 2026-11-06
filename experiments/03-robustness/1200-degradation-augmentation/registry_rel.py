"""Reliability of 400-row dev differences, estimated from the registry itself (no Public): split the 400 dev rows into random halves A/B;
true-effect variance across variants = cov(dA, dB); shrinkage of a 400-row estimate = var(true)/var(observed); winner's curse = the replicated
half-B effect of the variant that wins on half A, relative to its half-A effect. Controls with shuffled labels excluded. usage: registry_rel.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
Z = np.load(_repo_io('registry.npz')); names = Z['names']; R = Z['R']; keep = np.array(['perm' not in n for n in names]); R, names = R[keep], names[keep]
rng = np.random.default_rng(0); fam = {'slot': np.array([n.startswith('slot') for n in names]), 'partner': np.array([n.startswith('partner') for n in names])}
fam['all'] = np.ones(len(names), bool)
obs = R.mean(1); se2 = R.var(1) / 400
for f, msk in fam.items():
    cv, rep, slopes = [], [], []
    for _ in range(500):
        p = rng.permutation(400); a, b = R[msk][:, p[:200]].mean(1), R[msk][:, p[200:]].mean(1)
        cv.append(np.cov(a, b)[0, 1]); w = np.argmin(a); rep.append(b[w] / a[w]); slopes.append(np.polyfit(a, b, 1)[0])
    vt = max(np.mean(cv), 0); vo = obs[msk].var(); print(f'{f:8s} n={msk.sum():3d}: var(true) {vt:.3f}  var(obs400) {vo:.3f}  mean SE^2 {se2[msk].mean():.3f}  -> shrinkage of a 400-row dev difference {vt / vo:.2f}')
    print(f'          half-split: slope(B on A) {np.mean(slopes):.2f}; winner on half A keeps {np.mean(rep):.2f} of its effect on half B (median {np.median(rep):.2f})')
