"""Noisy-row DTA-partner share multiplier K (rest of 061 proportionally): dev plain, dev non-stretch, LM-dev noisy dMSE vs 061 (train only)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, nsgain as N
y, g, C, c61 = N.y, N.g, N.C, N.c61; w = np.load(_repo_io('lm_weights.npy'))
aw = 0.256; part = (C['ridge'] + C['logr']) / 2; other = (c61 - aw * part) / (1 - aw)
for K in (1, 1.25, 1.5, 1.75, 2, 2.5, 3):
    a = np.where(g, (1 - aw * K) * other + aw * K * part, c61); info, _ = N.decomp('', a, quiet=True)
    dm = (w * ((a - y) ** 2 - (c61 - y) ** 2))[:, g].mean(0); dp = ((a - y) ** 2 - (c61 - y) ** 2)[:, g].mean(0)
    print(f'K {K:4}: share {aw * K:.3f} | plain {N.rm(c61) - N.rm(a):+.4f} (noisy dMSE {dp.mean():+.2f}) | non-stretch {info:+.4f} | LM noisy dMSE {dm.mean():+.2f} (SE {dm.std() / np.sqrt(g.sum()):.2f})')
