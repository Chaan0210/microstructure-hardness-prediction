"""Per-grain axis-aligned size definitions (bounding box) vs rotation-invariant ones, aggregated per image. Label-free.
Masks: zero-shot masks-v3 (noise <= train median) / ft-H300 1-view (noisier). Grains touching the image border are excluded for bbox-based
definitions (truncated) and kept for counts. usage: bbox_feats.py SPLIT -> eda/bbox-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from skimage import measure
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
split = sys.argv[1]
Mc = np.load(_repo_path(E, f'grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); cpos = {str(i): k for k, i in enumerate(Mc['ids'])}
Mn = np.load(_repo_path(E, f'push-300/ftseg-ft-H300-{split}-noisy-n1-masks.npz'), allow_pickle=True); npos = {str(i): k for k, i in enumerate(Mn['ids'])}
F = np.load(_repo_path(E, f'grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
Ftr = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))
nz = F['X'][:, fk.index('noise')]
KEYS = ['bb_max_n', 'bb_geo_n', 'bb_mean_n', 'bb_w_n', 'bb_h_n', 'eqd_n', 'feret_n', 'bb_max_a', 'bb_geo_a', 'eqd_a', 'hp_bbmax_n', 'hp_bbgeo_n', 'hp_eqd_n', 'hp_bbmax_a', 'hp_eqd_a', 'n_int']
X = []
for n, i in enumerate(ids):
    m = (Mn['masks'][npos[i]] if nz[n] > thr else Mc['masks'][cpos[i]]).astype(int)
    pr = [p for p in measure.regionprops(m) if p.area >= 12]
    inn = [p for p in pr if p.bbox[0] > 0 and p.bbox[1] > 0 and p.bbox[2] < m.shape[0] and p.bbox[3] < m.shape[1]]
    if len(inn) < 10: X.append([np.nan] * len(KEYS)); continue
    w = np.array([p.bbox[3] - p.bbox[1] for p in inn], float); h = np.array([p.bbox[2] - p.bbox[0] for p in inn], float); A = np.array([p.area for p in inn], float)
    eqd = np.sqrt(4 * A / np.pi); fer = np.array([p.feret_diameter_max for p in inn]); bmax = np.maximum(w, h); bgeo = np.sqrt(w * h); bmean = (w + h) / 2
    aw = A / A.sum()
    X.append([bmax.mean(), bgeo.mean(), bmean.mean(), w.mean(), h.mean(), eqd.mean(), fer.mean(), (aw * bmax).sum(), (aw * bgeo).sum(), (aw * eqd).sum(),
              (bmax ** -0.5).mean(), (bgeo ** -0.5).mean(), (eqd ** -0.5).mean(), (aw * bmax ** -0.5).sum(), (aw * eqd ** -0.5).sum(), len(inn)])
(_repo_path(HERE, 'eda')).mkdir(exist_ok=True)
np.savez(_repo_path(HERE, f'eda/bbox-{split}.npz'), ids=np.array(ids), X=np.array(X), keys=np.array(KEYS)); print('done', np.array(X).shape)
