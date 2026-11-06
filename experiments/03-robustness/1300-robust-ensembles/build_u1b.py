"""Replace the three u1-derived production columns (u_ph, u_ph_hard, u_ph_f32) by the boundary-free-trained u1b U-Net's values, for real train/test
rows (feat-<v>-anghcor{split}) and degraded-copy rows (synthrows-<set>-<v>-anghcor). Outputs feat-<v>-anghcorB-{train,test}.npz (push-1300) and
synthrows-<set>-<v>-anghcorB.npz (push-1300). usage: build_u1b.py SETS(comma)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H4 = Path(_repo_io('../push-400')); H12 = Path(_repo_io('../push-1200')); V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; COLS = ['u_ph', 'u_ph_hard', 'u_ph_f32']
def ub(split, set_=''):
    U = np.load(_repo_path(H4, f'unet-u1b-{split}{set_}.npz'), allow_pickle=True); k = [str(x) for x in U['keys']]; return {str(i): U['X'][j, [k.index(c) for c in COLS]] for j, i in enumerate(U['ids'])}
for split in ('train', 'test'):
    m = ub(split)
    for v in V:
        F = dict(np.load(_repo_io(f'../push-800/feat-{v}-anghcor-{split}.npz'), allow_pickle=True)); keys = [str(x) for x in F['keys']]; X = F['X'].astype(float).copy()
        X[:, [keys.index(c) for c in COLS]] = np.array([m[str(i)] for i in F['ids']]); F['X'] = X; np.savez(_repo_io(f'feat-{v}-anghcorB-{split}.npz'), **F)
for s in sys.argv[1].split(','):
    m = ub('synth', f'train_{s}')
    for v in V:
        S = dict(np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True)); keys = [str(x) for x in S['keys']]; X = S['X'].astype(float).copy()
        X[:, [keys.index(c) for c in COLS]] = np.array([m[str(i)] for i in S['ids']]); S['X'] = X; np.savez(_repo_io(f'synthrows-{s}-{v}-anghcorB.npz'), **S)
print('built')
