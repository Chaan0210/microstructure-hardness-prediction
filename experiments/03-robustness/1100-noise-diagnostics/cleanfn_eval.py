"""Clean-function transfer partner: TabPFN-3.5 trained on clean+mid rows only (dev-<v>-anghcor-cleanfn.npz, 4 U-Net variants averaged).
Noisy rows (>= 12 gray) get y ~ a + lam * g_clean, (a, lam) fitted per partition on the noisy rows of the OTHER folds (their g_clean values are
out-of-sample for every clean model and never used a noisy label; the evaluated fold's labels are not used). Reported as a partner on top of c054.
usage: cleanfn_eval.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sys.path.insert(0, str(_repo_path(E, 'push-900')))
from blend_dev import load, rmse, parts, partner, align_dev
B = load(9); Y = B['y']; FOLD = B['fold']; ids = B['ids']
S9 = np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True); NZ = np.nan_to_num(S9['noise'], nan=0)
V = ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z')
gc = np.mean([align_dev(_repo_path(H, f'dev-{v}-anghcor-cleanfn.npz')) for v in V if (_repo_path(H, f'dev-{v}-anghcor-cleanfn.npz')).exists()], 0)
ga = np.mean([align_dev(_repo_path(E, f'push-800/dev-{v}-anghcor.npz')) for v in V], 0)     # the 044 slot (all rows), same 4 variants
noisy = NZ >= 12; bands = {'clean': NZ < 9.5, 'mid': (NZ >= 9.5) & (NZ < 12), 'noisy': noisy}
def bandrm(m): return ' '.join(f'{n} {np.sqrt(((m[:, b] - Y[b]) ** 2).mean()):.2f}' for n, b in bands.items())
print('044 slot all-rows TabPFN ', bandrm(ga)); print('clean-only TabPFN raw    ', bandrm(gc))
cal = gc.copy(); cal2 = gc.copy()
for k in range(3):
    for f in range(5):
        trn = noisy & (FOLD[k] != f); van = noisy & (FOLD[k] == f)
        A = np.c_[np.ones(trn.sum()), gc[k, trn]]; c = np.linalg.lstsq(A, Y[trn], rcond=None)[0]; cal[k, van] = c[0] + c[1] * gc[k, van]
        A2 = np.c_[np.ones(trn.sum()), gc[k, trn], ga[k, trn]]; c2 = np.linalg.lstsq(A2, Y[trn], rcond=None)[0]; cal2[k, van] = c2[0] + c2[1] * gc[k, van] + c2[2] * ga[k, van]
        if k == 0 and f == 0: print('fold0 calibration a, lam', c.round(3), ' joint', c2.round(3))
print('clean-only calibrated    ', bandrm(cal)); print('joint calibrated (gc,ga) ', bandrm(cal2))
for tag, m in [('cleanfn calibrated', cal), ('cleanfn joint-cal', cal2)]:
    partner(B, m, tag, shares=(0.05, 0.1, 0.15, 0.2, 0.3))
    mn = np.where(noisy, m, B['c054'])          # partner only on noisy rows
    partner(B, mn, tag + ' (noisy rows only)', shares=(0.1, 0.2, 0.3, 0.4))
