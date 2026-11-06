"""Robustness of the residual-CNN correction (candidate 082 = 076 + round4 aux-noise correction) on dev, original partition (the only partition the
round4 CNN was cross-fitted on): gain on the reproduced 076 OOF, image bootstrap CI, removal of the k largest contributors, stretch vs non-stretch
decomposition on noisy rows, per-fold gains. Reads only; writes work/promo/robust-082.json."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, csv, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; ME = HERE.parents[1]; E = _RepoPath(__file__).resolve().parents[4]
B = np.load(_repo_path(ME, 'repro/components/c14b-orig.npz')); y = B['y']; nz = B['nz_dev']; fail = B['fail_dev'][0]; b76 = B['dev'][0]
sid = [str(i) for i in np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]
D = np.load(_repo_path(E, 'rmse-research-20261015/round4-full073/noise-calibration/aux-oof.npz'), allow_pickle=True); p = {str(i): j for j, i in enumerate(D['ids'])}; ix = [p[i] for i in sid]
corr = (D['addition'] - D['baseline'])[ix]; fold = D['fold'][ix]; g = (nz >= 12) & ~fail
assert np.all(corr[~g] == 0) or np.abs(corr[~g]).max() < 1e-12, 'correction should touch gated rows only'
new = b76 + corr; se0, se1 = (b76 - y) ** 2, (new - y) ** 2; d = se1 - se0
rm = lambda a, m=slice(None): float(np.sqrt(((a - y)[m] ** 2).mean()))
rng = np.random.RandomState(0); bs = np.array([d[rng.randint(0, 400, 400)].mean() for _ in range(10000)])
o = np.argsort(d)   # most beneficial first
res = dict(rmse_076=rm(b76), rmse_082=rm(new), gain=rm(b76) - rm(new), dMSE=float(d.mean()), dMSE_CI95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           P_better=float((bs < 0).mean()), noisy_rmse=[rm(b76, g), rm(new, g)], n_improved=int((d[g] < 0).sum()), n_worse=int((d[g] > 0).sum()))
res['share_of_net_gain_top'] = {k: float(d[o[:k]].sum() / d.sum()) for k in (5, 10, 20)}
res['dMSE_after_removing_top_k_contributors'] = {}
for k in (5, 10, 20):
    keep = np.ones(400, bool); keep[o[:k]] = False; dk = d[keep]; bk = np.array([dk[rng.randint(0, keep.sum(), keep.sum())].mean() for _ in range(4000)])
    res['dMSE_after_removing_top_k_contributors'][k] = dict(dMSE=float(dk.mean()), P_better=float((bk < 0).mean()))
x = b76[g] - b76[g].mean(); s = float((x @ corr[g]) / (x @ x)); ns = new.copy(); ns[g] = b76[g] + corr[g] - s * x
res['stretch_coef'] = s; res['non_stretch_gain'] = rm(b76) - rm(ns); res['slope_y_on_pred_noisy'] = [float(np.polyfit(b76[g], y[g], 1)[0]), float(np.polyfit(new[g], y[g], 1)[0])]
res['per_fold_gain'] = {int(f): rm(b76, fold == f) - rm(new, fold == f) for f in range(5)}
res['winsorized_5_95_dMSE'] = float(np.clip(d, *np.percentile(d, [5, 95])).mean())
print(json.dumps(res, indent=1)); json.dump(res, open(_repo_path(HERE, 'robust-082.json'), 'w'), indent=1)
