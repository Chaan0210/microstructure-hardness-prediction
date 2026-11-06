"""Label-free trial of segmentation variants on low-coverage train images (coverage < 0.6)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np
from pathlib import Path
from PIL import Image
from cellpose import models, utils
from skimage import restoration, filters, transform
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
F = np.load(_repo_path(R, 'features-train.npz')); keys = list(F['keys'])
cov = np.nan_to_num(F['X'][:, keys.index('coverage')], nan=0)
low = [F['ids'][i] for i in np.argsort(cov)[:30:4]]  # 8 spread across the low-coverage tail
model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(R, 'weights/cpsam')))
def noise(f):
    hh = (f[0::2, 0::2] - f[0::2, 1::2] - f[1::2, 0::2] + f[1::2, 1::2]) / 2; return np.median(np.abs(hh)) / 0.6745
def run(x, **kw): m, _, _ = model.eval(x, **kw); return m
V = {
 'loose': lambda f: run(f, flow_threshold=0.8, cellprob_threshold=-2),
 'loose_cp-4': lambda f: run(f, flow_threshold=1.0, cellprob_threshold=-4),
 'gauss1.5': lambda f: run(filters.gaussian(f, 1.5), flow_threshold=0.8, cellprob_threshold=-2),
 'nlm': lambda f: run(restoration.denoise_nl_means(f, h=0.8 * noise(f), sigma=noise(f), patch_size=5, patch_distance=6, fast_mode=True), flow_threshold=0.8, cellprob_threshold=-2),
 'up2x_gauss1': lambda f: transform.resize(run(transform.resize(filters.gaussian(f, 1.0), (512, 512), order=3), flow_threshold=0.8, cellprob_threshold=-2), (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32),
}
res = {}; rows = []
for i in low:
    f = np.array(Image.open(_repo_path(P, f'data/train/{i}.png'))).astype(np.float32) / 255
    row = [np.stack([(f * 255).astype(np.uint8)] * 3, -1)]
    for name, fn in V.items():
        m = fn(f); c = float((m > 0).mean()); n = int(len(np.unique(m)) - 1)
        res.setdefault(name, []).append((c, n))
        rgb = np.stack([(f * 255).astype(np.uint8)] * 3, -1).copy()
        for o in utils.outlines_list(m): rgb[o[:, 1], o[:, 0]] = [255, 40, 40]
        row.append(rgb)
    rows.append(np.concatenate(row, 1)); print(i, {k: v[-1] for k, v in res.items()}, flush=True)
Image.fromarray(np.concatenate(rows[:4], 0)).save(_repo_path(R, 'lowcov-trial-a.png')); Image.fromarray(np.concatenate(rows[4:], 0)).save(_repo_path(R, 'lowcov-trial-b.png'))
print(json.dumps({k: dict(mean_cov=float(np.mean([c for c, _ in v])), mean_n=float(np.mean([n for _, n in v]))) for k, v in res.items()}, indent=1))
