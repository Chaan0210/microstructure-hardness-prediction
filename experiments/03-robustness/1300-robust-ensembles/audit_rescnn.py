"""Audit of round4 residual-CNN OOF (original partition, 400 rows): gain by band, stretch vs non-stretch on noisy rows (slope of the change on the
baseline's centred prediction), label-matched (LM) noisy dMSE, and calibration slope of y on prediction. Controls: context (no image), stats, cnn, aux."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
import nsgain as N
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
sid = list(np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']); sid = [str(i) for i in sid]
nz = N.nz; lw = np.load(_repo_io('lm_weights.npy')); lw0 = lw[0] if lw.ndim == 2 else lw
for tag in ('aux', 'cnn', 'context', 'stats'):
    D = np.load(_repo_io(f'../rmse-research-20261015/round4-full073/noise-calibration/{tag}-oof.npz'), allow_pickle=True); ids = [str(i) for i in D['ids']]
    j = [sid.index(i) for i in ids]; y = np.array([lab[i] for i in ids]); z = nz[j]; w = lw0[j]; g = z >= 12; fail = N.fail[0][j]
    b = D['baseline']; rm = lambda a, m=slice(None): np.sqrt(((a - y)[m] ** 2).mean())
    for arm in ('addition', 'replacement'):
        a = D[arm]; d = a - b; x = b[g] - b[g].mean(); s = (x @ d[g]) / (x @ x); ns = a.copy(); ns[g] = b[g] + d[g] - s * x
        dm = ((a - y) ** 2 - (b - y) ** 2); dl = (w * dm)[g].mean(); slope_b = np.polyfit(b[g], y[g], 1)[0]; slope_a = np.polyfit(a[g], y[g], 1)[0]
        print(f'{tag:7s} {arm:11s} gain {rm(b) - rm(a):+.4f} | clean<12 {rm(b, ~g) - rm(a, ~g):+.4f} noisy {rm(b, g) - rm(a, g):+.4f} | noisy stretch coef {s:+.3f}, '
              f'non-stretch gain {rm(b) - rm(ns):+.4f} | LM noisy dMSE {dl:+.2f} (plain {dm[g].mean():+.2f}) | noisy slope y~pred {slope_b:.2f} -> {slope_a:.2f}')
