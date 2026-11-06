from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, numpy as np, torch
sys.path.insert(0, _repo_io(str(_repo_path(_REPO_ROOT, 'experiments/02-microstructure/0300-cellpose-finetuning'))))
import os; os.chdir(_repo_io(str(_repo_path(_REPO_ROOT, 'experiments/02-microstructure/0300-cellpose-finetuning'))))
import harness
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
for n in [320, 900]:
    X = np.random.randn(n, 46); y = np.random.randn(n); Xv = np.random.randn(80, 46)
    t = time.time(); m = harness.tabpfn_new('v3.5', 0, sys.argv[1]); m.fit(X, y); p = m.predict(Xv); print(n, sys.argv[1], round(time.time() - t, 1), 's', flush=True)
