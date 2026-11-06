"""Side-by-side: clean train image | learned translation (trans-k1) | realdeg2 deg3 copy, and real noisy train images (crops 128 px, same scale)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = Path(__file__).resolve().parent; P0 = _RepoPath(__file__).resolve().parents[4]; U = _repo_path(H.parents[1], 'push-400')
D = np.load(_repo_path(H, 'trans-k1.npz')); ids = [str(i) for i in D['ids']]
def ng(a): a = a.astype(float); hh = (a[::2, ::2] - a[::2, 1::2] - a[1::2, ::2] + a[1::2, 1::2]) / 2; return float(np.median(np.abs(hh)) / .6745)
tr = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data/train.csv')))]
noisy = [i for i in tr if ng(np.asarray(Image.open(_repo_path(P0, 'data/train', f'{i}.png')))) >= 12][:4]
pick = [ids[k] for k in (3, 60, 150, 220)]
fig, ax = plt.subplots(4, 4, figsize=(12, 12))
for r, i in enumerate(pick):
    a = np.asarray(Image.open(_repo_path(P0, 'data/train', f'{i}.png'))); b = D['images'][0][ids.index(i)]; c = np.asarray(Image.open(_repo_path(U, 'synth/train_deg3', f'{i}.png')))
    nzi = np.asarray(Image.open(_repo_path(P0, 'data/train', f'{noisy[r]}.png')))
    for col, (im, t) in enumerate(((a, f'clean {i}'), (b, f'learned noisy (noise {ng(b):.1f})'), (c, f'realdeg2 deg3 ({ng(c):.1f})'), (nzi, f'real noisy {noisy[r]} ({ng(nzi):.1f})'))):
        ax[r, col].imshow(im[64:192, 64:192], cmap='gray', vmin=60, vmax=220); ax[r, col].set_title(t, fontsize=8); ax[r, col].axis('off')
plt.tight_layout(); plt.savefig(_repo_path(H, 'compare_cyc.png'), dpi=110); print('saved')
