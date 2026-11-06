"""Reliability-gated blending on top of 053: component weights depend on log N (grain count proxy) and pixel noise.
w_k(x) = w053_k + sum_j c_kj * z_j(x), z = standardized covariates; penalty lam*|c|^2; fitted by least squares on dev OOF.
Honest estimate: repeated random half-splits of the 400 ids (fit on one half of ids, all 3 partitions; evaluate on the other half).
usage: python gate_dev.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900'))
from blend_dev import load, rmse, parts
B = load(3); y = B['y']; fail = B['fail']
D = np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)
noise = np.nan_to_num(D['noise'], nan=np.nanmedian(D['noise'])); N = np.nan_to_num(D['Nproxy'], nan=np.nanmedian(D['Nproxy']))
C = np.stack([B['p021'], B['g'], B['ridge'], B['logridge'], np.where(fail, B['p021'], B['set']), B['gmm']], 0)   # [6,3,400]
import os
KS = [int(v) for v in os.environ.get("KS", "0,2,3,4,5").split(",")]
names = ['p021', 'g', 'ridge', 'logridge', 'set', 'gmm']
w0 = np.array([0.16, 0.384, 0.128, 0.128, 0.05, 0.15])
base = np.einsum('k,kpn->pn', w0, C); base = np.where(fail, B['p021'], base)
print('053 dev', rmse(base, y))
lz = np.log(N); lz = (lz - lz.mean()) / lz.std(); nz = (noise - noise.mean()) / noise.std()
COV = {'logN': [lz], 'noise': [nz], 'logN+noise': [lz, nz], 'logN+noise+int': [lz, nz, lz * nz]}
def design(Z, rows):
    # features: d_k * z_j for k in components (sum-to-one: shift weight from base), i.e. pred = base + sum c_kj z_j (C_k - base)
    cols = [z[rows] * (C[k][:, rows] - base[:, rows]) for k in KS for z in Z]
    return np.stack(cols, -1)   # [3, n, K]
def fit(Z, rows, lam):
    X = design(Z, rows).reshape(-1, len(Z) * len(KS)); r = (y[rows][None] - base[:, rows]).reshape(-1)
    return np.linalg.solve(X.T @ X + lam * len(r) * np.eye(X.shape[1]), X.T @ r)
def pred(Z, rows, c): return base[:, rows] + design(Z, rows) @ c
rng = np.random.RandomState(0); splits = [rng.permutation(400) for _ in range(300)]
for cname, Z in COV.items():
    for lam in (0.0, 1.0, 10.0, 100.0):
        G = []
        for idx in splits:
            A, Bh = idx[:200], idx[200:]
            A = A[~fail[0, A]]; c = fit(Z, A, lam)
            G.append(((base[:, Bh] - y[Bh]) ** 2).mean() - ((np.where(fail[:, Bh], base[:, Bh], pred(Z, Bh, c)) - y[Bh]) ** 2).mean())
        G = np.array(G); call = fit(Z, np.where(~fail[0])[0], lam); full = np.where(fail, base, base + design(Z, np.arange(400)) @ call)
        print(f'{cname:16s} lam {lam:6.1f}  honest dMSE {G.mean():+.3f} (sd {G.std():.3f}, P>0 {(G > 0).mean():.2f})  in-sample {rmse(full, y):.4f} parts {np.round(parts(base, y) - parts(full, y), 3)}')
    c = fit(Z, np.where(~fail[0])[0], 1.0).reshape(len(KS), len(Z)); print("   coefs (lam 1):", {names[k]: np.round(c[i], 3).tolist() for i, k in enumerate(KS)})
