from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, time, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, _repo_io('../grain-cellpose-220'))
from pathlib import Path
from PIL import Image
from skimage.restoration import denoise_nl_means
from degrade import degrade
from seglib import seg, seg_up2
from features_v3 import extract_v3
P = Path(_repo_io('../..'))
F = np.load(_repo_io('../grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
nz = np.load(_repo_io('noise_train.npy')); src = np.argsort(nz)[:80]
rng = np.random.RandomState(7)
out = {v: np.full((len(src), len(keys)), np.nan) for v in ['clean', 'deg', 'nlm', 'up2', 'nlm_up2']}
t0 = time.time()
def feats(im, m):
    f = extract_v3(im, m); return np.array([f.get(k, np.nan) if f else np.nan for k in keys])
for n, k in enumerate(src):
    im = np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png')))
    d = degrade(im, rng, blur=1.0, noise=17, contrast=0.85)
    out['clean'][n] = feats(im, seg(im))
    out['deg'][n] = feats(d, seg(d))
    dn = np.clip(np.round(denoise_nl_means(d.astype(np.float32) / 255, h=0.8 * 17 / 255, sigma=17 / 255, patch_size=5, patch_distance=6, fast_mode=True) * 255), 0, 255).astype(np.uint8)
    out['nlm'][n] = feats(dn, seg(dn))
    out['up2'][n] = feats(d, seg_up2(d))
    out['nlm_up2'][n] = feats(dn, seg_up2(dn))
    if (n + 1) % 10 == 0: print(n + 1, round(time.time() - t0), flush=True)
np.savez(_repo_io('fidelity.npz'), src=src, keys=np.array(keys), **out)
