"""Disagreement-gated DTA share on noisy rows: pred = o + (b1 + b2*|p-o|/s) * (p - o), o = 061 without the DTA partners (renormalised),
p = mean of the two DTA partners, s = fixed scale (dev pooled sd of p-o). 061 = b1 0.256, b2 0. Fitted by least squares on dev noisy rows of the
training folds (honest: weights from 4 folds, applied to the 5th, per partition); also LM-weighted fit. Train-only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, nsgain as N
y, g, C, c61, fold = N.y, N.g, N.C, N.c61, N.fold; lw = np.load(_repo_io('lm_weights.npy'))
aw = 0.256; p = (C['ridge'] + C['logr']) / 2; o = (c61 - aw * p) / (1 - aw); d = p - o; s = d[:, g].std()
def fitb(mask, wts, quad=True):
    Z = np.c_[d[mask], d[mask] * np.abs(d[mask]) / s] if quad else d[mask][:, None]; t = (y[None].repeat(3, 0) - o)[mask]
    W = wts[mask]; A = Z * W[:, None]; return np.linalg.lstsq(A.T @ Z, A.T @ t, rcond=None)[0]
def apply(b, quad=True): return o + (b[0] + (b[1] * np.abs(d) / s if quad else 0)) * d
for wnm, W in (('plain', np.ones((3, 400))), ('LM', np.tile(lw, (3, 1)))):
    for quad in (False, True):
        out = c61.copy(); bs = []
        for pp in range(3):
            for k in range(5):
                tr = np.zeros((3, 400), bool); tr[pp] = (fold[pp] != k) & g; b = fitb(tr, W, quad); bs.append(b); va = (fold[pp] == k) & g; out[pp, va] = apply(b, quad)[pp, va]
        bfull = fitb(np.tile(g, (3, 1)), W, quad); info, _ = N.decomp('', out, quiet=True)
        dm = (lw * ((out - y) ** 2 - (c61 - y) ** 2))[:, g].mean(0)
        print(f'fit {wnm:5s} {"quadratic gate" if quad else "constant share"}: full-dev b {np.round(bfull, 3)} (fold range b1 {min(b[0] for b in bs):.2f}..{max(b[0] for b in bs):.2f}) | honest dev {N.rm(out):.4f} ({N.rm(c61) - N.rm(out):+.4f}) non-stretch {info:+.4f} LM dMSE {dm.mean():+.2f} (SE {dm.std() / np.sqrt(g.sum()):.2f})')
np.save(_repo_io('dgate_s.npy'), s)
