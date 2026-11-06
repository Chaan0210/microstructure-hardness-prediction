"""Zero-shot (cpsam, seglib.seg = masks-v3 rule) segmentation + Study900 object extraction (extract_objects.extract) for the label-free degraded
copies of clean train images -> objbank-<SET>.npz in the object-bank format (for the gmm partner). usage: sims_objects.py SET"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, glob, numpy as np
from pathlib import Path
from PIL import Image
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_path(E, 'push-300'))); sys.path.insert(0, str(_repo_path(E, 'grain-cellpose-220'))); sys.path.insert(0, str(_repo_path(E, 'probabilistic-grains-900')))
import seglib
from extract_objects import extract, OBJECT_KEYS
st = sys.argv[1]; files = sorted(glob.glob(str(_repo_path(E, f'push-400/synth/train_{st}/*.png')))); ids = [Path(f).stem for f in files]
objs, weights, globs, offsets, gkeys = [], [], [], [0], None
for n, f in enumerate(files):
    im = np.asarray(Image.open(_repo_io(f))); m = seglib.seg(im); X, w, g = extract(im, m)
    objs.append(X); weights.append(w); offsets.append(offsets[-1] + len(w)); gkeys = gkeys or list(g); globs.append([g[k] for k in gkeys])
    if n % 50 == 0: print(st, n, flush=True)
np.savez_compressed(_repo_path(H, f'objbank-{st}.npz'), ids=np.array(ids), offsets=np.array(offsets), objects=np.concatenate(objs), object_keys=np.array(OBJECT_KEYS),
                    weights=np.concatenate(weights), global_X=np.array(globs), global_keys=np.array(gkeys)); print('done', st, offsets[-1])
