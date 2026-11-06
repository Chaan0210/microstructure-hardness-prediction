"""CV TabPFN on v3tta + extra feature block(s); paired with cached base OOF (same folds)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np, torch; torch.set_num_threads(2)
from harness import *
X, keys, ids = load_feats('features-v3tta'); y = labels(ids); nz = np.load(_repo_io('noise_train.npy'))
blocks = []
for spec in sys.argv[1].split(','):
    D = np.load(_repo_io(spec), allow_pickle=True); assert [str(i) for i in D['train_ids']] == ids; blocks.append(D['train'])
mode = sys.argv[2] if len(sys.argv) > 2 else 'concat'
E = np.hstack(blocks)
if mode == 'concat': XX = np.hstack([X, E])
elif mode == 'only': XX = E
report('base v3tta (cached)', np.load(_repo_io('oof_base_tabpfn.npy')), y, nz)
O = cv_predict(lambda: tabpfn(8, 0, 'mps'), XX, y, reps=(0, 1, 2)); report(f'{mode} {sys.argv[1]}', O, y, nz)
np.save(_repo_io(f'oof_{mode}_{sys.argv[1].replace(",", "+").replace(".npz", "")}.npy'), O)
