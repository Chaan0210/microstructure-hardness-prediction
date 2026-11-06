"""4-view segmentation TTA for noisy images (noise > median train noise; label-free rule). Extra views: transpose, flipud, rot90.
Each view uses the masks-v3 rule (cp -2, re-run cp -3 if coverage < 0.95). Features are computed per view (view image + view mask) and averaged later."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, time, numpy as np
from pathlib import Path
from PIL import Image
from cellpose import models
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
Ftr = np.load(_repo_path(R, 'features-v3-train.npz')); thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))
model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(R, 'weights/cpsam')))
VIEWS = {'T': lambda a: np.ascontiguousarray(a.T), 'UD': lambda a: np.ascontiguousarray(a[::-1]), 'R90': lambda a: np.ascontiguousarray(np.rot90(a))}
def seg(im):
    m, _, _ = model.eval(im.copy(), flow_threshold=0.8, cellprob_threshold=-2.0)
    if (m > 0).mean() < 0.95:
        m2, _, _ = model.eval(im.copy(), flow_threshold=1.0, cellprob_threshold=-3.0)
        if (m2 > 0).mean() > (m > 0).mean(): m = m2
    return m.astype(np.uint16)
for split in ['train', 'test']:
    out = _repo_path(R, f'masks-tta-{split}.npz')
    if out.exists(): continue
    F = np.load(_repo_path(R, f'features-v3-{split}.npz')); ids = F['ids']; nz = F['X'][:, list(F['keys']).index('noise')]
    sel = np.where(nz > thr)[0]; res = {v: np.zeros((len(sel), 256, 256), np.uint16) for v in VIEWS}; t0 = time.time()
    for n, k in enumerate(sel):
        im = np.array(Image.open(_repo_path(P, f'data/{split}/{ids[k]}.png')))
        for v, fn in VIEWS.items(): res[v][n] = seg(fn(im))
        if (n + 1) % 25 == 0: print(json.dumps(dict(split=split, done=n + 1, n=len(sel), sec=round(time.time() - t0))), flush=True)
    np.savez_compressed(_repo_io(out), ids=ids[sel], noise_threshold=thr, **res); print('DONE', split, len(sel), flush=True)
