"""Paired dev comparison of two c14_blend outputs (same Study900 rows/partitions): RMSE, per band, per partition, image bootstrap P(new better),
and test-prediction change. usage: c14_compare.py OLD.npz NEW.npz [NEW2.npz ...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
A = np.load(_repo_io(sys.argv[1])); y = A['y']; nz = A['nz_dev']; rng = np.random.RandomState(0); BI = [rng.randint(0, 400, 400) for _ in range(4000)]
rm = lambda a, m=slice(None): float(np.sqrt(((a - y)[:, m] ** 2).mean()))
bands = {'clean<9.5': nz < 9.5, 'mid': (nz >= 9.5) & (nz < 12), 'noisy>=12': nz >= 12}
print(f'{"OLD":12s} dev {rm(A["dev"]):.4f} ' + ' '.join(f'{k} {rm(A["dev"], m):.3f}' for k, m in bands.items()))
for f in sys.argv[2:]:
    B = np.load(_repo_io(f)); d = ((B['dev'] - y) ** 2 - (A['dev'] - y) ** 2).mean(0)
    print(f'{f.split("/")[-1][:12]:12s} dev {rm(B["dev"]):.4f} ({rm(A["dev"]) - rm(B["dev"]):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in np.sqrt(((A['dev'] - y) ** 2).mean(1)) - np.sqrt(((B['dev'] - y) ** 2).mean(1))) + ') '
          + ' '.join(f'{k} {rm(B["dev"], m):.3f}' for k, m in bands.items()) + f' | P(better) {np.mean([d[b].mean() < 0 for b in BI]):.3f} | test change RMS vs OLD {np.sqrt(((B["test"] - A["test"]) ** 2).mean()):.3f}')
    for k in ('tab', 'ridge', 'logr', 'gmm'):
        print(f'    comp {k:5s} dev RMSE old {rm(np.where(A["fail_dev"], A["dev_cnn"], A["dev_" + k])):.3f} new {rm(np.where(B["fail_dev"], B["dev_cnn"], B["dev_" + k])):.3f}')
