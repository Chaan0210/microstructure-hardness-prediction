"""Label-free diagnostic of U-Net readouts on real train images: structure is independent of the noise preset, so an unbiased readout has the same
distribution on clean (noise < 9.5) and noisy (>= 12) rows. Reports mean shift (noisy - clean, in clean SD units) and SD ratio for key columns,
per net. usage: unet_diag.py NET[,NET...]   (push-400 unet4-<NET>-train.npz or unet2-<NET>-train.npz)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
E = _RepoPath(__file__).resolve().parents[2]
nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255; cl = nz < 9.5; no = nz >= 12; mid = (nz >= 9.5) & (nz < 12)
COLS = ['m_ph', 'm_ph_hard', 'm_la_mean', 'm_la_std', 'm_el_pix', 'm_el_glob', 'm_la_f32', 'z_f25', 'z_f50']
for net in sys.argv[1].split(','):
    f = _repo_path(E, f'push-400/unet4-{net}-train.npz'); f = f if f.exists() else _repo_path(E, f'push-400/unet2-{net}-train.npz')
    U = np.load(_repo_io(f), allow_pickle=True); k = [str(x) for x in U['keys']]; X = U['X']
    out = []
    for c in COLS:
        if c not in k: continue
        v = X[:, k.index(c)]; s = v[cl].std()
        out.append(f'{c} shift mid {((v[mid].mean() - v[cl].mean()) / s):+.2f} noisy {((v[no].mean() - v[cl].mean()) / s):+.2f} sdratio {v[no].std() / s:.2f}')
    print(net, '|', ' ; '.join(out))
