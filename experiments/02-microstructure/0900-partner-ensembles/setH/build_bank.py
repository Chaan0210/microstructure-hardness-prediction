"""Object bank like probabilistic-grains-900/extract_objects.py, but with the production noisy-row segmentation: rows with features-v3 noise above the
train median use the fine-tuned ft-H300 1-view masks (push-300/ftseg-ft-H300-<split>-noisy-n1-masks.npz); the rest keep the zero-shot masks-v3.
Label-free per-image measurements. usage: build_bank.py SPLIT -> object-bank-<split>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[3]; P = _RepoPath(__file__).resolve().parents[4]
sys.path.insert(0, str(_repo_path(E, 'probabilistic-grains-900'))); from extract_objects import extract, OBJECT_KEYS
split = sys.argv[1]
rows = list(csv.DictReader(open(_repo_path(P, 'data', 'train.csv' if split == 'train' else 'sample_submission.csv')))); ids = [r['ID'] for r in rows]
Mc = np.load(_repo_path(E, f'grain-cellpose-220/masks-v3-{split}.npz')); cpos = {str(i): k for k, i in enumerate(Mc['ids'])}
Mn = np.load(_repo_path(E, f'push-300/ftseg-ft-H300-{split}-noisy-n1-masks.npz'), allow_pickle=True); npos = {str(i): k for k, i in enumerate(Mn['ids'])}
F = np.load(_repo_path(E, f'grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); fk = list(F['keys']); fpos = {str(i): k for k, i in enumerate(F['ids'])}
Ftr = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))
objs, weights, globs, offsets, src, gkeys = [], [], [], [0], [], None
for n, i in enumerate(ids):
    noisy = F['X'][fpos[i], fk.index('noise')] > thr
    if noisy: assert i in npos, i
    m = Mn['masks'][npos[i]] if noisy else Mc['masks'][cpos[i]]; src.append('H300' if noisy else 'v3')
    X, w, g = extract(np.asarray(Image.open(_repo_path(P, 'data', split, f'{i}.png'))), m); objs.append(X); weights.append(w); offsets.append(offsets[-1] + len(w))
    gkeys = gkeys or list(g); globs.append([g[k] for k in gkeys])
out = dict(ids=np.array(ids), offsets=np.array(offsets), objects=np.concatenate(objs), object_keys=np.array(OBJECT_KEYS), weights=np.concatenate(weights),
           global_X=np.array(globs), global_keys=np.array(gkeys), mask_source=np.array(src))
if split == 'train': out['y'] = np.array([float(r['hardness']) for r in rows])
np.savez_compressed(_repo_path(H, f'object-bank-{split}.npz'), **out); print(split, len(ids), 'grains', offsets[-1], 'H300 rows', src.count('H300'))
