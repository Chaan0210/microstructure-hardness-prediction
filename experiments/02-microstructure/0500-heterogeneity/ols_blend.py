"""Fold-honest test: OLS structure formula on clean rows (noise < GATE gray), fitted on outer-train clean rows only, mixed at fixed
weights with the 035 dev OOF. Reports clean-row and pooled dev RMSE per partition.
usage: ols_blend.py [GATE_GRAY=9.5] [FEATSET=f7]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, '.')
import resid035 as R
gate = float(sys.argv[1]) if len(sys.argv) > 1 else 9.5; fs = sys.argv[2] if len(sys.argv) > 2 else 'f7'
SETS = {'f7': ['fd_area', 'pore_frac_m', 'logasp_aw', 'S_align', 'logNeff', 'het4_px', 'bd_frac'],
        'f6': ['fd_area', 'pore_frac_m', 'logasp_aw', 'S_align', 'logNeff', 'het4_px'],
        'f5': ['fd_area', 'pore_frac_m', 'logasp_aw', 'S_align', 'logNeff'],
        'f7c': ['fd_area', 'pore_frac_m', 'logasp_aw', 'S_align', 'logNeff', 'het4_c', 'bd_frac'],
        'f8': ['fd_area', 'pore_frac_m', 'logasp_aw', 'S_align', 'logNeff', 'het4_px', 'bd_frac', 'la_aw'],
        'f7b': ['fd_area', 'pore_frac_m', 'logasp_aw', 'S_align', 'logNeff', 'het8_px', 'bd_frac']}
cols = SETS[fs]
S = np.load(_repo_io('struct-train.npz'), allow_pickle=True); sk = [str(k) for k in S['keys']]; spos = {str(i): k for k, i in enumerate(S['ids'])}
ids, y, B, G, Pp = R.load(); nz = R.noise(); n = np.array([nz[i] for i in ids]) * 255
Xs = S['X'][[spos[i] for i in ids]][:, [sk.index(c) for c in cols]]
clean = (n < gate) & ~np.isnan(Xs).any(1)
D = np.load(_repo_io('../push-300/dev-u1dmavg-v35.npz')); ref = list(ids)
W = [0.0, 0.2, 0.3, 0.4, 0.5]
rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
print(f'gate {gate} set {fs} clean rows {clean.sum()}/{len(ids)}')
se_all = {w: [] for w in W}
for k, part in enumerate(R.PARTS):
    I = [str(i) for i in D[part + '_ids']]; ix = {i: j for j, i in enumerate(I)}; f = D[part + '_fold'][[ix[i] for i in ref]]
    ols = np.full(len(ids), np.nan)
    for q in range(5):
        tr = (f != q) & clean; va = (f == q) & clean
        A = np.c_[np.ones(tr.sum()), Xs[tr]]; c, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        ols[va] = np.c_[np.ones(va.sum()), Xs[va]] @ c
    b = B[k]; out = {}
    for w in W:
        p = b.copy(); p[clean] = (1 - w) * b[clean] + w * ols[clean]; se_all[w].append((p - y) ** 2)
        out[w] = (rm((p - y)[clean]), rm(p - y))
    print(part, 'ols alone clean %.3f' % rm((ols - y)[clean]), ' '.join(f'w{w}: clean {a:.3f} all {b_:.4f}' for w, (a, b_) in out.items()))
print('pooled', ' '.join(f'w{w}: {np.sqrt(np.mean(se_all[w], 0).mean()):.4f}' for w in W))
