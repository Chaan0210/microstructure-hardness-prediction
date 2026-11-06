"""D3: hardness-label and rendering-robust structure distributions, clean copy sources (noise < train median, 249) vs real noise >= 12 rows (161),
all 500 train rows; also split into dev-400 / holdout-100. Writes d3.json."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, numpy as np
from pathlib import Path
from scipy import stats
W = Path(__file__).resolve().parents[2]; E = _RepoPath(__file__).resolve().parents[4]; P = _RepoPath(__file__).resolve().parents[5]
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]; y = np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))])
nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255; src = nz < np.median(nz); noisy = nz >= 12; mid = ~src & ~noisy
dev = set(str(i) for i in np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']); isdev = np.array([i in dev for i in ids])
F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
X = F['X'][[pos[i] for i in ids]].astype(float)
def comp(a, b):
    return dict(n=[int(len(a)), int(len(b))], mean=[float(a.mean()), float(b.mean())], sd=[float(a.std(ddof=1)), float(b.std(ddof=1))],
                q05_q95=[[float(np.percentile(a, 5)), float(np.percentile(a, 95))], [float(np.percentile(b, 5)), float(np.percentile(b, 95))]],
                ks_p=float(stats.ks_2samp(a, b).pvalue), levene_p=float(stats.levene(a, b).pvalue), welch_p=float(stats.ttest_ind(a, b, equal_var=False).pvalue))
out = {'labels_all500': comp(y[src], y[noisy]), 'labels_dev400': comp(y[src & isdev], y[noisy & isdev]), 'labels_holdout100': comp(y[src & ~isdev], y[noisy & ~isdev]),
       'labels_mid_band': comp(y[src], y[mid])}
rng = np.random.RandomState(0); a, b = y[src], y[noisy]; r = [np.std(rng.choice(b, len(b)), ddof=1) / np.std(rng.choice(a, len(a)), ddof=1) for _ in range(4000)]
out['sd_ratio_noisy_over_src_CI95'] = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
out['robust_structure'] = {k: dict(src=float(np.nanmedian(X[src, keys.index(k)])), noisy=float(np.nanmedian(X[noisy, keys.index(k)])),
                                   ks_p=float(stats.ks_2samp(X[src, keys.index(k)][~np.isnan(X[src, keys.index(k)])], X[noisy, keys.index(k)][~np.isnan(X[noisy, keys.index(k)])]).pvalue))
                           for k in ['px_pore_d', 'ang_c4', 'ang_s4', 'orient_coherence', 'm_el_glob']}
json.dump(out, open(_repo_path(Path(__file__).parent, 'd3.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
