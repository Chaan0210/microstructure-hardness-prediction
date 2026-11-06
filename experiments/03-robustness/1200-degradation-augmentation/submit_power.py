"""How clearly can one Public submission (300 rows) read each candidate's change vs 061? Dev rows give the per-row paired dSE = e_new^2 - e_old^2;
Public SE of dMSE ~ sd(dSE)/sqrt(300). Expected Public dMSE taken as dev dMSE x {1, 0.4} (0.4 = historical Public/dev ratio, rough).
Also pairwise steps (064 vs 062, 065 vs 064, 066 vs 065). usage: submit_power.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, glob, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
B = load(3); D = np.load(_repo_io('dev065.npz')); y, nz, fail = D['y'], D['nz'], D['fail']; g12 = (nz >= 12) & ~fail
c53 = 0.16 * B['p021'] + 0.384 * np.mean([D[f'b{s}'] for s in ('38', '39', '44')], 0) + 0.128 * B['ridge'] + 0.128 * B['logridge'] + 0.05 * B['set'] + 0.15 * B['gmm']; c53 = np.where(B['fail'], B['p021'], c53)
def part(t): S = np.load(_repo_io(f'sr-{t}.npz')); return c53 + 0.128 * g12 * ((S['ridge'] - B['ridge']) + (S['logridge'] - B['logridge']))
c61, c62 = part('A-d1234n12'), part('A-d1to7n12-wal')
ts = sum(D[f's{s}'] - D[f'b{s}'] for s in ('38', '39', '44')); c64 = c62 + 0.128 * g12 * ts; c65 = c62 + 0.128 * ts
c66 = c65.copy()
for k, pn in enumerate(['original', '20261031', '20261101']):
    ec = np.mean([np.load(_repo_io(f))['oof'] for f in sorted(glob.glob(f'cnn-ctrl-{pn}-s*.npz'))], 0); ed = np.mean([np.load(_repo_io(f))['oof'] for f in sorted(glob.glob(f'cnn-dta-{pn}-s*.npz'))], 0); c66[k] += 0.16 * g12 * (ed - ec)
assert abs(np.sqrt(((c65 - y) ** 2).mean()) - 9.5842) < 1e-3
C = dict(c53=c53, c61=c61, c62=c62, c64=c64, c65=c65, c66=c66); rm = lambda a: np.sqrt(((a - y) ** 2).mean())
print(f'Public 061 = 8.9831 (MSE {8.9831 ** 2:.2f}); dRMSE ~ dMSE / (2*8.98)')
for a, b in [('c61', 'c62'), ('c61', 'c64'), ('c61', 'c65'), ('c61', 'c66'), ('c62', 'c64'), ('c64', 'c65'), ('c65', 'c66')]:
    d = ((C[b] - y) ** 2 - (C[a] - y) ** 2).ravel(); m = d.mean(); se = d.std() / np.sqrt(300)
    print(f'{b[1:]} vs {a[1:]}: dev {rm(C[a]):.4f}->{rm(C[b]):.4f} dMSE {m:+.3f} | Public SE(dMSE) {se:.3f} = {se / 17.97:.4f} RMSE | z at x1 {m / se:+.2f}, at x0.4 {0.4 * m / se:+.2f}')
