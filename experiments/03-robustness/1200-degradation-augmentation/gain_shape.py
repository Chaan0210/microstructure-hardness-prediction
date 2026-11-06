"""Where does each change's dev gain sit? Noisy rows (>=12, not failed) split by the 053 blend prediction: extreme 20% (largest |pred - median|)
vs central 80%, and low/high halves. Changes: 057 (partners sr-A-d23), 061 (sr-A-d1234n12), 062 (sr-A-d1to7n12-wal), 065 (TA slots), 066 (CNN delta).
Also the 'stretch' of each change on noisy rows: slope of delta on (053 pred - mean). usage: gain_shape.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, glob, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
B = load(3); D = np.load(_repo_io('dev065.npz')); y, nz, fail = D['y'], D['nz'], D['fail']; m = (nz >= 12) & ~fail
c53 = 0.16 * B['p021'] + 0.384 * np.mean([D[f'b{s}'] for s in ('38', '39', '44')], 0) + 0.128 * B['ridge'] + 0.128 * B['logridge'] + 0.05 * B['set'] + 0.15 * B['gmm']; c53 = np.where(B['fail'], B['p021'], c53)
def part(tag):
    S = np.load(_repo_io(f'sr-{tag}.npz')); return c53 + 0.128 * m * ((S['ridge'] - B['ridge']) + (S['logridge'] - B['logridge']))
c62 = part('A-d1to7n12-wal'); c65 = D['c']
c66 = c65.copy()
for k, pn in enumerate(['original', '20261031', '20261101']):
    ec = np.mean([np.load(_repo_io(f))['oof'] for f in sorted(glob.glob(f'cnn-ctrl-{pn}-s*.npz'))], 0); ed = np.mean([np.load(_repo_io(f))['oof'] for f in sorted(glob.glob(f'cnn-dta-{pn}-s*.npz'))], 0)
    c66[k] = c65[k] + 0.16 * m * (ed - ec)
cands = [('057', c53, part('A-d23')), ('061', c53, part('A-d1234n12')), ('062', c53, c62), ('065 vs 062', c62, c65), ('066 vs 065', c65, c66)]
for nm, a, b in cands:
    ks = [0] if 'orig' in nm else [0, 1, 2]; out = []
    for k in ks:
        p = c53[k]; dev_ = np.abs(p - np.median(p[m])); thr = np.quantile(dev_[m], 0.8); ext = m & (dev_ >= thr); cen = m & (dev_ < thr)
        lo, hi = m & (p < np.median(p[m])), m & (p >= np.median(p[m]))
        g = lambda mm: ((a[k] - y)[mm] ** 2).sum() - ((b[k] - y)[mm] ** 2).sum()
        d = (b[k] - a[k])[m]; pc = (p - p[m].mean())[m]; slope = (d * pc).sum() / (pc * pc).sum()
        out.append((g(m), g(ext), g(cen), g(lo), g(hi), slope))
    o = np.mean(out, 0); print(f'{nm:18s} SSE gain noisy {o[0]:8.0f}: extreme20 {o[1]:7.0f} ({o[1] / o[0]:.0%}) central80 {o[2]:7.0f} | low {o[3]:7.0f} high {o[4]:7.0f} | stretch slope {o[5]:+.3f}')
