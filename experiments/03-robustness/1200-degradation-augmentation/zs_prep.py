"""ZS-net step 1 (label-free; TRAIN images only): for every clean train image (pixel noise < 9.5 gray) store the image, the per-pixel area of
the containing zero-shot grain (masks-v3), the pixel class map (push-1000/geom_prep.classes: 0 matrix, 1 second phase, 2 pore, 3 boundary)
and K degraded copies (push-1000/geom_prep.degrade: 55% calibrated noisy preset realdeg2, generic noise/blur/contrast/shading otherwise).
Hardness labels and test images are not used. -> zs-prep.npz, zs-bank-u8.npy [n, K, 256, 256]
usage: zs_prep.py K"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from multiprocessing import Pool
from PIL import Image
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_path(E, 'push-1000'))); sys.path.insert(0, str(_repo_path(E, 'push-900')))
from geom_prep import classes, degrade, ng
def _bank(args):
    k, a, m, K, pool = args; r = np.random.RandomState(70_000 + k)
    return np.stack([degrade(a.astype(np.float64), m.astype(np.int64), r, pool).astype(np.uint8) for _ in range(K)])
if __name__ == '__main__':
    K = int(sys.argv[1])
    ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data/train.csv')))]
    M = np.load(_repo_path(E, 'grain-cellpose-220/masks-v3-train.npz')); mpos = {str(i): k for k, i in enumerate(M['ids'])}
    imgs = {i: np.asarray(Image.open(_repo_path(P0, f'data/train/{i}.png'))) for i in ids}; nz = {i: ng(imgs[i]) for i in ids}
    clean = [i for i in ids if nz[i] < 9.5]; Is, Ms, Ts, LA = [], [], [], []
    for i in clean:
        m = M['masks'][mpos[i]].astype(np.int64); T, G = classes(imgs[i].astype(np.float64), m)
        area = np.bincount(m.ravel(), minlength=m.max() + 1).astype(np.float64); la = np.log(np.maximum(area[m], 1.0)); la[m == 0] = 0
        Is.append(imgs[i]); Ms.append(m.astype(np.int32)); Ts.append(T); LA.append(la.astype(np.float32))
    np.savez(_repo_path(H, 'zs-prep.npz'), ids=np.array(clean), images=np.array(Is), masks=np.array(Ms), T=np.array(Ts), LA=np.array(LA))
    pool = np.array([v for v in nz.values() if v >= 9.5]); print('clean images', len(clean), 'noise pool', len(pool), flush=True)
    with Pool(7) as p: bank = p.map(_bank, [(k, Is[k], Ms[k], K, pool) for k in range(len(clean))])
    np.save(_repo_path(H, 'zs-bank-u8.npy'), np.array(bank)); print('bank', np.array(bank).shape, flush=True)
