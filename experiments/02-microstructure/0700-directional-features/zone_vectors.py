"""16 zone (64 px = 32 map px) state vectors per image from z1 U-Net maps (train): [mean logA, mean logAR, c4, s4, z_f25, z_f50]. Label-free."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys
split = sys.argv[1] if len(sys.argv) > 1 else 'train'
M = np.load(_repo_io(f'../push-400/unet4maps-z1-{split}.npz'), allow_pickle=True); ids = [str(i) for i in M['ids']]; maps = M['maps']
Z = np.zeros((len(ids), 16, 6), np.float32)
for n in range(len(ids)):
    m = maps[n].astype(np.float64); g = m[0] + m[1]; la = m[4]; c2, s2 = m[5], m[6]; z = m[7]
    r = lambda a: a.reshape(4, 32, 4, 32).sum((1, 3)).ravel()
    W = r(g) + 1e-6
    C = r(g * c2) / W; S = r(g * s2) / W; v = np.sqrt(C * C + S * S) + 1e-9
    Z[n, :, 0] = r(g * la) / W; Z[n, :, 1] = r(g * np.hypot(c2, s2)) / W; Z[n, :, 2] = (C * C - S * S) / v; Z[n, :, 3] = 2 * C * S / v
    Z[n, :, 4] = r(g / (1 + np.exp(-(z - 0.25) / 0.02))) / W; Z[n, :, 5] = r(g / (1 + np.exp(-(z - 0.5) / 0.02))) / W
np.savez(_repo_io(f'zonevec-{split}.npz'), ids=np.array(ids), Z=Z, keys=np.array(['logA', 'logAR', 'c4', 's4', 'zf25', 'zf50']))
print(split, Z.shape, Z.reshape(-1, 6).mean(0).round(3), Z.reshape(-1, 6).std(0).round(3))
