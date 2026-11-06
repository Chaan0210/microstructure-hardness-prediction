"""gmm partner (Study900 image_gmm__x98_augmented: Ridge on X98 + per-image GMM phase readout) with the aligned X98 (qm) on dev; swap into 054.
Also the variant with the GMM readout columns aligned the same way. usage: gmm_qm.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
src = open(_repo_io('gmm_partner.py')).read().split("mode = sys.argv[1]")[0]; ns = {'__file__': str(Path('gmm_partner.py').resolve())}; exec(src, ns)
from blend_dev import load, rmse, parts
B = load(9); ids, y, FO = B['ids'], B['y'], B['fold']
S9 = Path(_repo_io('../probabilistic-grains-900')); Btr = np.load(_repo_path(S9, 'object-bank-train.npz'), allow_pickle=True)
def X_of(path): F = np.load(_repo_io(path), allow_pickle=True); pos = {str(i): k for k, i in enumerate(F['ids'])}; return F['X'][[pos[i] for i in ids]].astype(float)
Xq = X_of('feat-u1dmavgH2Z-anghcorqm-train.npz'); Gf = ns['gmm_feats'](Btr, ids)
P = np.zeros((3, 400))
for s in range(3):
    for f in range(5):
        tr, va = FO[s] != f, FO[s] == f; X = np.c_[Xq, Gf]; P[s, va], _ = ns['fit_predict'](X[tr], y[tr], X[va], 19001 + 101 * s + f)
np.savez(_repo_io('gmm-qm-dev.npz'), oof=P)
fail, p = B['fail'], B['p021']; R = np.load(_repo_io('ridge-qm.npz'))
def c54(r, l, g): c51 = 0.2 * p + 0.48 * B['g'] + 0.16 * r + 0.16 * l; return np.where(fail, p, 0.77 * c51 + 0.08 * B['set'] + 0.15 * g)
base = c54(B['ridge'], B['logridge'], B['gmm'])
for nm, args in (('gmm qm only', (B['ridge'], B['logridge'], P)), ('ridges qm', (R['ridge'], R['logridge'], B['gmm'])), ('all three qm', (R['ridge'], R['logridge'], P))):
    c = c54(*args); print(f'{nm:13s}: gmm alone {rmse(np.where(fail, p, args[2]), y):.3f}  054 {rmse(c, y):.4f} ({rmse(base, y) - rmse(c, y):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in parts(base, y) - parts(c, y)) + ')')
