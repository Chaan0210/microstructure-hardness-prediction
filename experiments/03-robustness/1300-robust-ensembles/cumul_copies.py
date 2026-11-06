"""Cumulant measurements (same measure() as cumul_ext.py) for the degraded copies deg..deg4 (all copies; filter noise >= 12 later).
maps: push-1200/synthmaps-train_<set>-u4z1.npz; noise: synthrows-<set>-u1dmavgH2Z-anghcor 'noise' x255. -> cumul-<set>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys
from pathlib import Path
from PIL import Image
exec(open(_repo_io('cumul_ext.py')).read().split("P = Path('../..')")[0])
P = Path(_repo_io('../..'))
for st in ('deg', 'deg2', 'deg3', 'deg4'):
    M = np.load(_repo_io(f'../push-1200/synthmaps-train_{st}-u4z1.npz')); mid = [str(i) for i in M['ids']]
    S = np.load(_repo_io(f'../push-1200/synthrows-{st}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); sk = [str(k) for k in S['keys']]; sids = [str(i) for i in S['ids']]
    nz = dict(zip(sids, np.nan_to_num(S['X'][:, sk.index('noise')].astype(float) * 255, nan=0)))
    Z = np.array([measure(np.asarray(Image.open(_repo_path(P, f'experiments/push-400/synth/train_{st}/{i}.png')).convert('L'), float), M['maps'][j], nz[i]) for j, i in enumerate(mid)])
    np.savez(_repo_io(f'cumul-{st}.npz'), ids=np.array(mid), X=Z); print(st, Z.shape, np.isfinite(Z).all(), flush=True)
