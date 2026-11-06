"""Swap the 044 TabPFN slot inside the c054 structure: g045 = mean(038, 039, 044 slots) (4 U-Net variants each); c054 = 0.77*(0.2*021 + 0.48*g + 0.16*ridge
+ 0.16*logRidge) + 0.08*set + 0.15*gmm. usage: eval_slot.py TAG [TAG ...]  (dev-<v>-anghcor<TAG>.npz in push-1100, or push-800 when TAG == '')"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sys.path.insert(0, str(_repo_path(E, 'push-900')))
from blend_dev import load, rmse, parts, align_dev
B = load(9); Y = B['y']
S9 = np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True); NZ = np.nan_to_num(S9['noise'], nan=0)
bands = {'clean': NZ < 9.5, 'mid': (NZ >= 9.5) & (NZ < 12), 'noisy': NZ >= 12}
V = ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z')
slot = lambda pat: np.mean([align_dev(pat.format(v=v)) for v in V], 0)
s38 = slot(str(_repo_path(E, 'push-300/dev-{v}-v35.npz'))); s39 = slot(str(_repo_path(E, 'push-600/dev-{v}-angh.npz'))); s44 = slot(str(_repo_path(E, 'push-800/dev-{v}-anghcor.npz')))
gchk = (s38 + s39 + s44) / 3; print('grain045 reproduced: max |diff|', np.abs(np.where(B['fail'], 0, gchk - B['g'])).max().round(6))
def c054(g):
    c51 = 0.2 * B['p021'] + 0.48 * g + 0.16 * B['ridge'] + 0.16 * B['logridge']; return np.where(B['fail'], B['p021'], 0.77 * c51 + 0.08 * B['set'] + 0.15 * B['gmm'])
ref = c054(B['g']); r0 = rmse(ref, Y)
def br(m): return ' '.join(f'{n} {np.sqrt(((m[:, b] - Y[b]) ** 2).mean()):.3f}' for n, b in bands.items())
print(f'base 044 slot alone {rmse(s44, Y):.4f} ({br(s44)}) | c054 {r0:.4f}')
for t in sys.argv[1:]:
    s = slot(str(_repo_path(H, 'dev-{v}-anghcor' + t + '.npz'))); g = (s38 + s39 + s) / 3; c = c054(g)
    print(f'{t:8s} slot alone {rmse(s, Y):.4f} ({br(s)}) | c054 {rmse(c, Y):.4f} gain {r0 - rmse(c, Y):+.4f} parts '
          + '/'.join(f'{v:+.3f}' for v in parts(ref, Y) - parts(c, Y)) + ' | band gain ' + ' '.join(
              f'{n} {np.sqrt(((ref[:, b] - Y[b]) ** 2).mean()) - np.sqrt(((c[:, b] - Y[b]) ** 2).mean()):+.3f}' for n, b in bands.items()), flush=True)
