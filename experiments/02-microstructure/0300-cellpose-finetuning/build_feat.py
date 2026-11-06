"""Build a feature npz: features-v3 base, noisy rows <- mean over views of FT npz, optional across-view std columns of key features.
usage: build_feat.py SPLIT OUT FT_NPZ [FT_NPZ_EXTRA_VIEWS ...] [--vstd]  (extra npz files are concatenated along the view axis)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, hashlib, numpy as np
args = [a for a in sys.argv[1:] if not a.startswith('--')]; vstd = '--vstd' in sys.argv
split, out, fts = args[0], args[1], args[2:]
F = np.load(_repo_io(f'../grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); X = F['X'].astype(float).copy(); keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(ids)}
Vs = [np.load(_repo_io(f), allow_pickle=True) for f in fts]
for V in Vs[1:]: assert [str(i) for i in V['ids']] == [str(i) for i in Vs[0]['ids']]
for V in Vs: assert [str(k) for k in V['keys']] == keys
VX = np.concatenate([V['X'] for V in Vs], 1); rows = np.array([pos[str(i)] for i in Vs[0]['ids']])
X[rows] = np.nanmean(VX, 1)
KK = ['sp_frac_grains', 'sp_count_frac', 'f_dark_20', 'f_dark_35', 'sizefield_range_s32', 'sizefield_std_s32', 'sizefield_std_s16', 'tile8_logA_std', 'grain_logA_std', 'grain_eqd_mean', 'aw_eqd_mean', 'n_obj', 'solidity_wmean', 'neighbors_std', 'ar_wmean', 'orient_coherence']
if vstd:
    J = [keys.index(k) for k in KK]; E = np.full((len(ids), len(J)), np.nan); E[rows] = np.nanstd(VX[:, :, J], 1)
    X = np.hstack([X, E]); keys = keys + [f'vstd_{k}' for k in KK]
np.savez(_repo_io(out), X=X, keys=np.array(keys), ids=np.array(ids)); print(out, X.shape, 'views', VX.shape[1], 'rows', len(rows))
sha = lambda p: hashlib.sha256(open(_repo_io(p), 'rb').read()).hexdigest()
base = f'../grain-cellpose-220/features-v3-{split}.npz'
json.dump(dict(file=out, sha256=sha(out), split=split, base=base, base_sha256=sha(base), rule='noisy-half rows (ids in the ftseg files) <- mean over all views; other rows = base',
               ftseg_sources=[dict(file=f, sha256=sha(f), views=int(np.load(_repo_io(f), allow_pickle=True)['X'].shape[1])) for f in fts], n_views=int(VX.shape[1]), n_rows_replaced=int(len(rows)),
               vstd=vstd, vstd_columns=[k for k in keys if k.startswith('vstd_')], columns=keys), open(_repo_io(out.replace('.npz', '.provenance.json')), 'w'), indent=1)
