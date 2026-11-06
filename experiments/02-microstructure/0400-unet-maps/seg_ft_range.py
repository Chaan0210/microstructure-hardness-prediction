"""Fine-tuned Cellpose-SAM segmentation (masks-v3 rule) + v3 features for images whose noise lies in a TRAIN noise-quantile range.
usage: seg_ft_range.py CKPT SPLIT QLO QHI [NVIEWS=4]
  selects images with train-quantile(QLO) < noise <= train-quantile(QHI) (thresholds from train features-v3 'noise' only)
output: ftr-<ckpt>-<split>-q<QLO>-<QHI>-n<NVIEWS>.npz : ids, X [n, views, keys], keys, masks [n, views, 256, 256] (resumable via .partial.npz)
views: id, T, UD, R90 (same as push-300 seg_ft.py); masks are stored in the VIEW frame.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_path(HERE.parent, 'push-300'))); sys.path.insert(0, str(_repo_path(HERE.parent, 'grain-cellpose-220')))
from PIL import Image
import seglib
from features_v3 import extract_v3
ckpt, split, qlo, qhi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]); nviews = int(sys.argv[5]) if len(sys.argv) > 5 else 4
from cellpose import models
seglib._model = models.CellposeModel(gpu=True, pretrained_model=ckpt)
P = _RepoPath(__file__).resolve().parents[3]
F = np.load(_repo_path(HERE.parent, f'grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
Ftr = np.load(_repo_path(HERE.parent, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); ntr = Ftr['X'][:, list(Ftr['keys']).index('noise')]
lo, hi = np.nanquantile(ntr, qlo), np.nanquantile(ntr, qhi)
nz = F['X'][:, keys.index('noise')]
sel = np.where((nz > lo) & (nz <= hi))[0]
V = [lambda a: a, lambda a: np.ascontiguousarray(a.T), lambda a: np.ascontiguousarray(a[::-1]), lambda a: np.ascontiguousarray(np.rot90(a))][:nviews]
name = f'ftr-{Path(ckpt).stem}-{split}-q{qlo:g}-{qhi:g}-n{nviews}'; part = _repo_path(HERE, f'{name}.partial.npz')
out = np.full((len(sel), nviews, len(keys)), np.nan); masks = np.zeros((len(sel), nviews, 256, 256), np.uint16); start = 0
if part.exists():
    Z = np.load(_repo_io(part), allow_pickle=True); start = int(Z['done']); out[:start] = Z['X'][:start]; masks[:start] = Z['masks'][:start]
print(json.dumps(dict(n=len(sel), lo=float(lo), hi=float(hi), resume=start)), flush=True); t0 = time.time()
for n, k in enumerate(sel):
    if n < start: continue
    im = np.array(Image.open(_repo_path(P, f'data/{split}/{ids[k]}.png')))
    for v, f in enumerate(V):
        x = f(im); m = seglib.seg(x); r = extract_v3(x, m); out[n, v] = [r.get(q, np.nan) if r else np.nan for q in keys]; masks[n, v] = m
    if (n + 1) % 10 == 0: np.savez(_repo_io(part), X=out, masks=masks, done=n + 1)
    if (n + 1) % 25 == 0: print(json.dumps(dict(done=n + 1, n=len(sel), sec=round(time.time() - t0))), flush=True)
np.savez_compressed(_repo_path(HERE, f'{name}.npz'), ids=np.array([ids[j] for j in sel]), X=out, keys=np.array(keys), masks=masks, lo=lo, hi=hi)
part.unlink(missing_ok=True); print('DONE', round(time.time() - t0), flush=True)
