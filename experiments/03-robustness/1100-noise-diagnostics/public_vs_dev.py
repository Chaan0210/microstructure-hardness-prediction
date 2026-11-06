"""Do past Public readings agree better with the dev score on clean+mid rows (or inverse-variance weighted) than with plain dev?
Dev OOF of each submitted structure reconstructed from saved component OOFs (Study900 ids, 3 partitions)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sys.path.insert(0, str(_repo_path(E, 'push-900')))
from blend_dev import load, align_dev
B = load(3); B9 = load(9); Y = B['y']; p = B['p021']; F = B['fail']
S9 = np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True); NZ = np.nan_to_num(S9['noise'], nan=0); noisy = NZ >= 12
V = ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z')
sl = lambda pat: np.mean([align_dev(pat.format(v=v)) for v in V], 0)
s38, s39, s44 = sl(str(_repo_path(E, 'push-300/dev-{v}-v35.npz'))), sl(str(_repo_path(E, 'push-600/dev-{v}-angh.npz'))), sl(str(_repo_path(E, 'push-800/dev-{v}-anghcor.npz')))
W = lambda a: np.where(F, p, a); g = (s38 + s39 + s44) / 3
c51 = W(0.2 * p + 0.48 * g + 0.16 * B['ridge'] + 0.16 * B['logridge'])
C = {'038': (W(0.25 * p + 0.75 * s38), 9.1817), '039': (W(0.25 * p + 0.75 * s39), 9.1950), '044': (W(0.25 * p + 0.75 * s44), 9.1987),
     '045': (W(0.2 * p + 0.8 * g), 9.1590), '048': (W(0.2 * p + 0.8 * (0.8 * g + 0.2 * B['ridge'])), 9.0998),
     '050': (W(0.2 * p + 0.8 * (0.6 * g + 0.4 * B['logridge'])), 9.0847),
     '053': (W(0.8 * c51 + 0.05 * B['set'] + 0.15 * B['gmm']), 9.0381)}
def sc(a, w): e2 = ((a - Y) ** 2).mean(0); return np.sqrt((w * e2).sum() / w.sum())
ws = {'plain': np.ones(400), 'clean+mid only': (~noisy).astype(float), 'noisy w0.45': np.where(noisy, 0.45, 1.0), 'noisy only': noisy.astype(float)}
pub = np.array([v[1] for v in C.values()])
for nm, w in ws.items():
    d = np.array([sc(v[0], w) for v in C.values()])
    print(f'{nm:15s} ' + ' '.join(f'{k} {x:.3f}' for k, x in zip(C, d)) + f' | spearman with Public {spearmanr(d, pub)[0]:+.3f} pearson {pearsonr(d, pub)[0]:+.3f}')
print('Public         ', ' '.join(f'{k} {x:.3f}' for k, x in zip(C, pub)))
