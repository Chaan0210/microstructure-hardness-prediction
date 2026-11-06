"""Train-only diagnostic: decompose each candidate's dev change vs 053 on noisy rows into the 057 direction u = c057 - c053 (effective K =
<d,u>/<u,u>; 058 = K 3 by construction) and the orthogonal rest. Public so far: 057 (K1) and 061 better, 059 (K3 minus stretch) and 064 worse.
usage: kdiag.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, glob, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
B = load(3); D = np.load(_repo_io('dev065.npz')); y, nz, fail = D['y'], D['nz'], D['fail']; m = (nz >= 12) & ~fail
c53 = 0.16 * B['p021'] + 0.384 * np.mean([D[f'b{s}'] for s in ('38', '39', '44')], 0) + 0.128 * B['ridge'] + 0.128 * B['logridge'] + 0.05 * B['set'] + 0.15 * B['gmm']; c53 = np.where(B['fail'], B['p021'], c53)
def part(t, base=c53, K=1.0): S = np.load(_repo_io(f'sr-{t}.npz')); return base + K * 0.128 * m * ((S['ridge'] - B['ridge']) + (S['logridge'] - B['logridge']))
c57, c61, c62 = part('A-d23'), part('A-d1234n12'), part('A-d1to7n12-wal'); c58 = part('A-d23', K=3)
ts = sum(D[f's{s}'] - D[f'b{s}'] for s in ('38', '39', '44')); c64 = c62 + 0.128 * m * ts; c65 = c62 + 0.128 * ts
cnn = np.zeros_like(c53)
for k, pn in enumerate(['original', '20261031', '20261101']):
    cnn[k] = 0.16 * m * (np.mean([np.load(_repo_io(f))['oof'] for f in sorted(glob.glob(f'cnn-dta-{pn}-s*.npz'))], 0) - np.mean([np.load(_repo_io(f))['oof'] for f in sorted(glob.glob(f'cnn-ctrl-{pn}-s*.npz'))], 0))
C = {'057': c57, '058 (K3)': c58, '061': c61, '062': c62, '064': c64, '065': c65, '061+cnn': c61 + cnn, '062+cnn': c62 + cnn, '065+cnn (066)': c65 + cnn}
u = (c57 - c53)[:, m].ravel(); rm = lambda a: np.sqrt(((a - y) ** 2).mean())
print(f'053 dev {rm(c53):.4f}')
for n, c in C.items():
    d = (c - c53)[:, m].ravel(); K = d @ u / (u @ u); res = d - K * u
    print(f'{n:15s} dev {rm(c):.4f}  K_eff {K:5.2f}  |d| {np.sqrt((d ** 2).mean()):.3f}  orth share {np.sqrt((res ** 2).mean() / (d ** 2).mean()):.2f}')
print('--- on the 061 line (same-weight replacements), gain shape vs 061 on noisy rows')
C2 = {'062': c62, '061+cnn': c61 + cnn, '062+cnn': c62 + cnn, 'sf0.5': part('A-d1to7n12-wal-sf0.5'), 'sf1': part('A-d1to7n12-wal-sf1'),
      'd1to8': part('A-d1to8n12-wal'), 'sf0.5+cnn': part('A-d1to7n12-wal-sf0.5') + cnn}
for n, c in C2.items():
    d = (c - c53)[:, m].ravel(); K = d @ u / (u @ u); out = []
    for k in range(3):
        p = c53[k]; dv = np.abs(p - np.median(p[m])); thr = np.quantile(dv[m], .8); ext, cen = m & (dv >= thr), m & (dv < thr)
        g = lambda mm: ((c61[k] - y)[mm] ** 2).sum() - ((c[k] - y)[mm] ** 2).sum(); out.append((g(m), g(ext), g(cen)))
    o = np.mean(out, 0); pp = np.sqrt(((c - y) ** 2).mean(1)) - np.sqrt(((c61 - y) ** 2).mean(1))
    print(f'{n:10s} dev {rm(c):.4f} (vs 061 {rm(c61) - rm(c):+.4f}; parts {np.round(-pp, 3)})  K {K:.2f} | SSE gain {o[0]:6.0f} extreme {o[1]:6.0f} central {o[2]:6.0f}')
print('--- 061 + TA slots (noise>=12)')
c61ta = c61 + 0.128 * m * ts
for n, c in {'061+TA': c61ta}.items():
    d = (c - c53)[:, m].ravel(); K = d @ u / (u @ u); out = []
    for k in range(3):
        p = c53[k]; dv = np.abs(p - np.median(p[m])); thr = np.quantile(dv[m], .8); ext, cen = m & (dv >= thr), m & (dv < thr)
        g = lambda mm: ((c61[k] - y)[mm] ** 2).sum() - ((c[k] - y)[mm] ** 2).sum(); out.append((g(m), g(ext), g(cen)))
    o = np.mean(out, 0); pp = np.sqrt(((c - y) ** 2).mean(1)) - np.sqrt(((c61 - y) ** 2).mean(1))
    print(f'{n:10s} dev {rm(c):.4f} (vs 061 {rm(c61) - rm(c):+.4f}; parts {np.round(-pp, 3)})  K {K:.2f} | SSE gain {o[0]:6.0f} extreme {o[1]:6.0f} central {o[2]:6.0f}')
