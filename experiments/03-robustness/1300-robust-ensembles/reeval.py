"""Re-evaluate push-1200 noisy-row variants vs 061 with the train-only robust criteria: non-stretch gain and label-matched dev (LM)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import glob, numpy as np, nsgain as N
from blend_dev import load
B = load(3); y, g, C, c61 = N.y, N.g, N.C, N.c61; w = np.load(_repo_io('lm_weights.npy'))
def lm(new):
    dm = (w * ((new - y) ** 2 - (c61 - y) ** 2))[:, g].mean(0); return dm.mean(), dm.std() / np.sqrt(g.sum())
def show(nm, new):
    info, _ = N.decomp(nm, new, quiet=True); a, s = lm(new); tot = N.rm(c61) - N.rm(new)
    print(f'{nm:34s} plain {tot:+.4f}  non-stretch {info:+.4f}  LM noisy dMSE {a:+.2f} (SE {s:.2f}, z {a / s:+.1f})')
for f in sorted(glob.glob('../push-1200/sr-A-*.npz')):
    t = f.split('sr-A-')[1][:-4]
    if t in ('d1234n12',): continue
    S = np.load(_repo_io(f)); Cx = dict(C); Cx['ridge'] = np.where(g, S['ridge'], B['ridge']); Cx['logr'] = np.where(g, S['logridge'], B['logridge']); show(f'partners {t}', N.blend(Cx))
for f in sorted(glob.glob('../push-1200/sg-*.npz')):
    S = np.load(_repo_io(f)); Cx = dict(C); Cx['gmm'] = np.where(g, S['oof'], C['gmm']); show(f'gmm {f.split("sg-")[1][:-4]}', N.blend(Cx))
import glob as G
cnn = np.zeros((3, 400))
for k, pn in enumerate(['original', '20261031', '20261101']):
    cnn[k] = np.mean([np.load(_repo_io(f))['oof'] for f in sorted(G.glob(f'../push-1200/cnn-dta-{pn}-s*.npz'))], 0) - np.mean([np.load(_repo_io(f))['oof'] for f in sorted(G.glob(f'../push-1200/cnn-ctrl-{pn}-s*.npz'))], 0)
show('CNN DTA delta (066-type)', np.where(g, c61 + 0.16 * cnn, c61))
