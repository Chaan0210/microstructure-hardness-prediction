"""Complete strict 055 on dev: the gmm partner (Study900 image_gmm__x98_augmented Ridge) with the X98 alignment fitted on the outer-training rows of
each fold only (ridge_qm_strict.aligned), combined with the strict TabPFN slots (dev-<v>-{qs,anghqs,anghcorqs}.npz) and strict Ridge partners
(ridge-qm-strict.npz) in the 054/055 structure. usage: gmm_qm_strict.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from pathlib import Path
src = open(_repo_io('gmm_partner.py')).read().split("mode = sys.argv[1]")[0]; ns = {'__file__': str(Path('gmm_partner.py').resolve())}; exec(src, ns)
rs = open(_repo_io('ridge_qm_strict.py')).read(); aligned_src = rs[rs.index('def rs(x):'):rs.index('r1 = np.zeros')]
from blend_dev import load, rmse, parts, align_dev
B = load(9); ids, y, FO, fail, p = B['ids'], B['y'], B['fold'], B['fail'], B['p021']
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
X0 = F['X'][[pos[i] for i in ids]].astype(float); j = keys.index('noise'); exec(aligned_src)
S9 = Path(_repo_io('../probabilistic-grains-900')); Gf = ns['gmm_feats'](np.load(_repo_path(S9, 'object-bank-train.npz'), allow_pickle=True), ids)
G = np.zeros((3, 400))
for k in range(3):
    for f in range(5):
        tr, va = FO[k] != f, FO[k] == f; Xa = np.c_[aligned(X0, tr), Gf]; G[k, va], _ = ns['fit_predict'](Xa[tr], y[tr], Xa[va], 19001 + 101 * k + f)
np.savez(_repo_io('gmm-qm-strict-dev.npz'), oof=G)
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
slot = lambda pat: np.mean([align_dev(pat.format(v)) for v in V], 0)
gs = np.mean([slot('dev-{}-qs.npz'), slot('dev-{}-anghqs.npz'), slot('dev-{}-anghcorqs.npz')], 0); RS = np.load(_repo_io('ridge-qm-strict.npz'))
W = lambda a: np.where(fail, p, a); S3 = B['sets'][:3].mean(0)
c53 = W(0.80 * B['c051'] + 0.05 * S3 + 0.15 * B['gmm'])
c55s = W(0.77 * (0.2 * p + 0.48 * gs + 0.16 * RS['ridge'] + 0.16 * RS['logridge']) + 0.08 * B['set'] + 0.15 * G)
c54 = B['c054']
print('gmm strict alone', round(rmse(W(G), y), 4), ' (in-sample qm 10.061, original 10.191)')
for nm, c in (('053', c53), ('054', c54), ('055 strict (complete)', c55s)):
    print(f'{nm:22s} dev {rmse(c, y):.4f}  parts ' + '/'.join(f'{v:.3f}' for v in parts(c, y)))
rng = np.random.default_rng(1); a2 = ((c53 - y) ** 2).mean(0); b2 = ((c55s - y) ** 2).mean(0); bs = rng.integers(0, 400, (10000, 400))
d = np.sqrt(a2[bs].mean(1)) - np.sqrt(b2[bs].mean(1)); print(f'complete strict 055 vs 053: {np.sqrt(a2.mean()) - np.sqrt(b2.mean()):+.4f}  95% CI [{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}]  partitions ' + '/'.join(f'{v:+.3f}' for v in parts(c53, y) - parts(c55s, y)))
