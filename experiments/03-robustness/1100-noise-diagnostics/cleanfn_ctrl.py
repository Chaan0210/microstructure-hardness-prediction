"""Controls for cleanfn_eval.py: is the noisy-row gain information from the clean-trained function, or just the slope stretch (047 family)?
All partners act on noisy rows only (c054 elsewhere); calibrations fitted on the other folds' noisy rows of the same partition."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sys.path.insert(0, str(_repo_path(E, 'push-900')))
from blend_dev import load, rmse, parts, align_dev
B = load(9); Y = B['y']; FOLD = B['fold']; c54 = B['c054']
S9 = np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True); NZ = np.nan_to_num(S9['noise'], nan=0); noisy = NZ >= 12
V = ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z')
gc = np.mean([align_dev(_repo_path(H, f'dev-{v}-anghcor-cleanfn.npz')) for v in V], 0); ga = np.mean([align_dev(_repo_path(E, f'push-800/dev-{v}-anghcor.npz')) for v in V], 0)
def calib(cols, slope1=False):
    out = c54.copy()
    for k in range(3):
        for f in range(5):
            trn = noisy & (FOLD[k] != f); van = noisy & (FOLD[k] == f)
            if slope1:   # intercept only on (y - x)
                x = cols[0][k]; out[k, van] = x[van] + np.mean(Y[trn] - x[trn]); continue
            A = np.c_[np.ones(trn.sum())] if not cols else np.c_[np.ones(trn.sum()), *[c[k, trn] for c in cols]]
            co = np.linalg.lstsq(A, Y[trn], rcond=None)[0]; out[k, van] = co[0] + sum(co[j + 1] * c[k, van] for j, c in enumerate(cols))
    return out
def rep(tag, m):
    d = m - c54; aopt = -np.mean(d * (c54 - Y)) / np.mean(d * d); best = c54 + aopt * d
    line = f'{tag:34s} noisy-rmse {np.sqrt(((m[:, noisy] - Y[noisy]) ** 2).mean()):.2f}  opt a {aopt:.2f} -> {rmse(best, Y):.4f}'
    for a in (0.2, 0.3, 0.4):
        b = c54 + a * d; line += f' | a{a} {rmse(c54, Y) - rmse(b, Y):+.4f} (' + '/'.join(f'{v:+.3f}' for v in parts(c54, Y) - parts(b, Y)) + ')'
    print(line, flush=True)
print('c054 noisy rmse', np.sqrt(((c54[:, noisy] - Y[noisy]) ** 2).mean()).round(3))
rep('stretch c054 (a+l*c054)', calib([c54]))
rep('stretch 044slot (a+l*ga)', calib([ga]))
rep('cleanfn a+l*gc', calib([gc]))
rep('cleanfn intercept only (l=1)', calib([gc], slope1=True))
rep('cleanfn + c054 (a+l1*gc+l2*c054)', calib([gc, c54]))
rep('raw gc (no calibration)', np.where(noisy, gc, c54))
# what slope does the in-fold fit choose for each?
for nm, x in [('c054', c54), ('ga', ga), ('gc', gc)]:
    sl = [np.polyfit(x[k, noisy], Y[noisy], 1)[0] for k in range(3)]; print(nm, 'noisy-row slope y~x', np.round(sl, 3), ' pred sd', x[:, noisy].std(1).round(2))
