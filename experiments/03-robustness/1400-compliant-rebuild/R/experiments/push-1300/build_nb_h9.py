"""Build the 073 'nb' sets (drop 6 intensity + 7 het cols) and the 076 'h9' sets (+ cumulant sigma-1 p34 and contrast) for the three
TabPFN slots (038 H2Z, 039 angh, 044 anghcor) x four U-Net variants, from an experiments tree.
usage: build_nb_h9.py EXPROOT OUTDIR   (EXPROOT holds push-500/700/800 feature sets and push-1300/cumul-{split}.npz)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
E, O = Path(sys.argv[1]), Path(sys.argv[2]); O.mkdir(parents=True, exist_ok=True)
DROP = ['coverage', 'dark_span_vs_noise', 'grain_I_std_mean', 'img_std', 'obj_I_cv', 'obj_I_spread_vs_noise', 'sizefield_plane_r2', 'sizefield_range_s32',
        'sizefield_ratio', 'sizefield_std_s16', 'sizefield_std_s32', 'tile8_logA_std', 'tile_logA_std']
SLOT = {'038': 'push-500/feat-{v}H2Z-{s}.npz', '039': 'push-700/feat-{v}H2Z-angh-{s}.npz', '044': 'push-800/feat-{v}H2Z-anghcor-{s}.npz'}
for s in ('train', 'test'):
    C = np.load(_repo_io(E / f'push-1300/cumul-{s}.npz'), allow_pickle=True); cid = [str(i) for i in C['ids']]
    for slot, pat in SLOT.items():
        for v in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4'):
            F = np.load(_repo_io(E / pat.format(v=v, s=s)), allow_pickle=True); keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]; X = F['X'].astype(float)
            pos = {i: j for j, i in enumerate(cid)}; H = C['X'][[pos[i] for i in ids]][:, [1, 4]]; hk = ['cumulant_sigma1.0_p34', 'cumulant_sigma1.0_contrast']
            keep = [j for j, k in enumerate(keys) if k not in DROP]; assert len(keep) == len(keys) - len(DROP)
            nbk = [keys[j] for j in keep]
            np.savez(_repo_io(O / f'feat-nb-{slot}-{v}-{s}.npz'), ids=F['ids'], X=X[:, keep], keys=np.array(nbk))
            np.savez(_repo_io(O / f'feat-h9-nb-{slot}-{v}-{s}.npz'), ids=F['ids'], X=np.c_[X[:, keep], H], keys=np.array(nbk + hk))
            np.savez(_repo_io(O / f'feat-h9-full-{slot}-{v}-{s}.npz'), ids=F['ids'], X=np.c_[X, H], keys=np.array(keys + hk))
    print('built', s, flush=True)
