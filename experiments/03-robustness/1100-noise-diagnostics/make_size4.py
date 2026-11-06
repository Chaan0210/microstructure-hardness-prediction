"""044-slot feature sets + 4 exact size columns (log_density, log_mean_ecd, log_aw_ecd, ecd_cv from push-1000 geom-exact, zero-shot masks) for clean
rows (pixel noise < 9.5); NaN elsewhere. Nulls: 4 N(0,1) columns with the same NaN pattern (draw s). Train split only (dev screening).
usage: make_size4.py  -> feat-<v>-anghcor{S4,R4a,R4b}-train.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
G = np.load(_repo_path(E, 'push-1000/geom-exact-train.npz'), allow_pickle=True); gpos = {str(i): k for k, i in enumerate(G['ids'])}; KG = [str(k) for k in G['keys']]
cols = ['log_density', 'log_mean_ecd', 'log_aw_ecd', 'ecd_cv']; XG = G['X'][:, [KG.index(c) for c in cols]].astype(float)
for v in ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z'):
    F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
    ex = np.array([i in gpos for i in ids]); S = np.full((len(ids), 4), np.nan); S[ex] = XG[[gpos[i] for i in ids if i in gpos]]
    for tag, A in [('S4', S)] + [(f'R4{c}', np.where(ex[:, None], np.random.RandomState(7700 + n).normal(size=(len(ids), 4)), np.nan)) for n, c in enumerate('ab')]:
        np.savez(_repo_path(H, f'feat-{v}-anghcor{tag}-train.npz'), ids=F['ids'], X=np.c_[F['X'].astype(float), A], keys=np.array(keys + [f'x_{c}' for c in cols]))
    print(v, 'exact rows', ex.sum())
