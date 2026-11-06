"""Realism of synthetic noisy copies vs real noisy train images, through the production u1 U-Net phase fraction (push-400/unet-u1-*.npz):
for each synthetic set: corr(u_ph_syn, u_ph_clean_source), sd ratio sd(syn)/sd(clean source), corr(u_ph_syn, y);
real noisy rows (pixel noise >= 12): sd ratio vs the clean population, corr with y. Also the ft cellpose n_obj ratio when features exist.
usage: realism.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np
from pathlib import Path
U = _repo_path(Path(__file__).resolve().parents[2], 'push-400'); P0 = _RepoPath(__file__).resolve().parents[4]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P0, 'data/train.csv')))}
def ld(f, key='u_ph'):
    d = np.load(_repo_io(f), allow_pickle=True); k = [str(x) for x in d['keys']]; return {str(i): float(v) for i, v in zip(d['ids'], d['X'][:, k.index(key)])}
T = ld(_repo_path(U, 'unet-u1-train.npz')); ids = list(T)
from PIL import Image
def ng(i):
    a = np.asarray(Image.open(_repo_path(P0, 'data/train', f'{i}.png')), float); hh = (a[::2, ::2] - a[::2, 1::2] - a[1::2, ::2] + a[1::2, 1::2]) / 2; return float(np.median(np.abs(hh)) / .6745)
NZ = {i: ng(i) for i in ids}; clean = [i for i in ids if NZ[i] < 9.5]; noisy = [i for i in ids if NZ[i] >= 12]
c = np.array([T[i] for i in clean]); n = np.array([T[i] for i in noisy])
print(f"real noisy (n={len(noisy)}): sd ratio {n.std() / c.std():.3f}  corr y {np.corrcoef(n, [lab[i] for i in noisy])[0, 1]:.3f}   (clean pop corr y {np.corrcoef(c, [lab[i] for i in clean])[0, 1]:.3f})")
for nm, f in (('deg', 'unet-u1-synth.npz'), ('deg2', 'unet-u1-synth2.npz'), ('deg3', 'unet-u1-synth3.npz'), ('cyc', 'unet-u1-synthcyc.npz')):
    if not (_repo_path(U, f)).exists(): print(nm, 'missing'); continue
    S = ld(_repo_path(U, f)); sid = [i for i in S if i in T]; s = np.array([S[i] for i in sid]); src = np.array([T[i] for i in sid]); y = np.array([lab[i] for i in sid])
    print(f'{nm:5s} (n={len(sid)}): corr(syn, clean src) {np.corrcoef(s, src)[0, 1]:.3f}  sd ratio {s.std() / src.std():.3f}  corr y {np.corrcoef(s, y)[0, 1]:.3f} (src {np.corrcoef(src, y)[0, 1]:.3f})')
