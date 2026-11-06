"""Band RMSE of the single-variant (u1dmavgH2Z) 044-slot TabPFN with degraded clean-row phase/size columns vs the undegraded run.
usage: eval_degph.py TAG [TAG ...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sys.path.insert(0, str(_repo_path(E, 'push-900')))
from blend_dev import load, align_dev
B = load(9); Y = B['y']; S9 = np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True); NZ = np.nan_to_num(S9['noise'], nan=0)
bands = {'clean': NZ < 9.5, 'mid': (NZ >= 9.5) & (NZ < 12), 'noisy': NZ >= 12}
def show(tag, g): print(f'{tag:8s} ' + ' '.join(f'{n} rmse {np.sqrt(((g[:, b] - Y[b]) ** 2).mean()):.3f} (MSE {((g[:, b] - Y[b]) ** 2).mean():6.1f})' for n, b in bands.items()))
show('base', align_dev(_repo_path(E, 'push-800/dev-u1dmavgH2Z-anghcor.npz')))
for t in sys.argv[1:]: show(t, align_dev(_repo_path(H, f'dev-u1dmavgH2Z-anghcor{t}.npz')))
