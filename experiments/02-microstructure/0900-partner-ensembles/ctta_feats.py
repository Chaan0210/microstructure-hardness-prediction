"""Clean-half segmentation TTA features: features-v3 (64 cols, zero-shot cpsam) recomputed on 3 extra views (masks-ctta-{split}.npz) and averaged with
the original view for the clean rows (noise <= train median). Noisy rows unchanged (they already carry the 4-view ft-a1s250 average).
Then rebuild the 038 / 039 / 044 variant sets: their v3 columns are replaced for clean rows only (checked equal to features-v3 first).
usage: ctta_feats.py SPLIT   -> features-v3ctta-{split}.npz, feat-<v>{H2Z,H2Z-angh,-anghcor}-ctta-{split}.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, warnings, numpy as np
from pathlib import Path
from PIL import Image
warnings.filterwarnings('ignore')
H = Path(__file__).resolve().parent; R = _repo_path(H.parent, 'grain-cellpose-220'); P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(R)); from features_v3 import extract_v3
split = sys.argv[1]
VIEWS = {'T': lambda a: np.ascontiguousarray(a.T), 'UD': lambda a: np.ascontiguousarray(a[::-1]), 'R90': lambda a: np.ascontiguousarray(np.rot90(a))}
F = np.load(_repo_path(R, f'features-v3-{split}.npz')); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float).copy(); pos = {str(i): k for k, i in enumerate(F['ids'])}
T = np.load(_repo_path(H, f'masks-ctta-{split}.npz')); tids = [str(i) for i in T['ids']]; VS = np.zeros((len(tids), len(keys)))
for n, i in enumerate(tids):
    im = np.array(Image.open(_repo_path(P, f'data/{split}/{i}.png'))); vals = [F['X'][pos[i]].astype(float)]
    for v, fn in VIEWS.items():
        f = extract_v3(fn(im), T[v][n]); vals.append(np.array([f.get(k, np.nan) if f else np.nan for k in keys], float))
    V = np.vstack(vals); X[pos[i]] = np.nanmean(V, 0); VS[n] = np.nanstd(V, 0)
    if (n + 1) % 100 == 0: print(split, n + 1, flush=True)
np.savez(_repo_path(H, f'features-v3ctta-{split}.npz'), ids=F['ids'], X=X, keys=np.array(keys), ctta_ids=np.array(tids), view_sd=VS)
sets = {'038': ('../push-500/feat-{v}H2Z-{s}.npz', 'feat-{v}H2Z-ctta-{s}.npz'), '039': ('../push-700/feat-{v}H2Z-angh-{s}.npz', 'feat-{v}H2Z-angh-ctta-{s}.npz'),
        '044': ('../push-800/feat-{v}-anghcor-{s}.npz', 'feat-{v}-anghcor-ctta-{s}.npz')}
for c, (src, dst) in sets.items():
    for v in ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z') if c == '044' else ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4'):
        S = np.load(_repo_path(H, src.format(v=v, s=split)), allow_pickle=True); sk = [str(k) for k in S['keys']]; SX = S['X'].astype(float).copy(); sids = [str(i) for i in S['ids']]
        cols = [k for k in keys if k in sk]; rows = [sids.index(i) for i in tids]
        old = SX[np.ix_(rows, [sk.index(k) for k in cols])]; ref = F['X'][[pos[i] for i in tids]][:, [keys.index(k) for k in cols]].astype(float)
        assert np.allclose(old, ref, equal_nan=True), (c, v, 'clean-row v3 columns differ from features-v3')
        SX[np.ix_(rows, [sk.index(k) for k in cols])] = X[[pos[i] for i in tids]][:, [keys.index(k) for k in cols]]
        np.savez(_repo_path(H, dst.format(v=v, s=split)), ids=S['ids'], X=SX, keys=S['keys']); print(c, v, split, 'replaced', len(cols), 'cols x', len(rows), 'rows', flush=True)
