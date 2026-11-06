from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, '.')
from deform_feats import matrix, KEYS
M0 = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz')); ids = [str(i) for i in M0['ids']]
Mn = np.load(_repo_io('../push-300/ftseg-ft-a1s250-train-noisy-n1-aug-masks.npz')); nid = [str(i) for i in Mn['ids']]
D0 = matrix(M0['masks']); Dn = matrix(Mn['masks'])
pos = {i: k for k, i in enumerate(ids)}; D = D0.copy(); D[[pos[i] for i in nid]] = Dn
np.savez(_repo_io('deform-train-zsclean-ftaugnoisy.npz'), ids=np.array(ids), X=D, keys=np.array(KEYS))
print('saved', D.shape, np.isnan(D).sum())
