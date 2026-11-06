"""H3 screen S3: TabPFN slot-044 members with +im_mean / +random column, production dev protocol (push-300/dev_eval.py logic,
original partition only). usage: tab_screen.py ARM VARIANT FS [FOLDS]   ARM in base|mean|rand, FS in nb|full
Writes h3/tab/{ARM}-{FS}-{VARIANT}.npz (OOF g for the 400 dev ids in the original-partition order)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
import sys
sys.dont_write_bytecode = True
import json
import time
import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
E = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_path(E, 'push-300')))
import harness  # noqa: E402

arm, variant, fs = sys.argv[1], sys.argv[2], sys.argv[3]
folds = [int(f) for f in sys.argv[4].split(',')] if len(sys.argv) > 4 else list(range(5))
device = os.environ.get('TAB_DEVICE', 'mps')
fpath = _repo_path(E, f'push-1300/abl/feat-h9phys-{fs}-044-{variant}-train.npz')
F = np.load(_repo_io(fpath), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
XA = F['X'].astype(float)
S = np.load(_repo_path(HERE, 'spec-train.npz'), allow_pickle=True)
assert [str(i) for i in S['ids']] == [str(i) for i in F['ids']]
if arm == 'mean':
    XA = np.c_[XA, S['X'][:, 0]]
elif arm == 'rand':
    XA = np.c_[XA, np.random.RandomState(20261017).standard_normal(len(XA))]
else:
    assert arm == 'base'
d = np.load(_repo_path(E, 'resolution-216/ensemble-OOF.npz'))
ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']
X = XA[[pos[i] for i in ids]]
out = _repo_path(HERE, f'tab/{arm}-{fs}-{variant}.npz')
out.parent.mkdir(exist_ok=True)
g = np.full(len(y), np.nan)
if out.exists():
    z = np.load(_repo_io(out)); g = z['g'].copy()
t0 = time.time()
for k in folds:
    tr, va = fold != k, fold == k
    ps = []
    for sd in (0, 1, 2):
        m = harness.tabpfn_new('v3.5', 100 * sd + k, device, None)
        m.fit(X[tr], y[tr]); ps.append(m.predict(X[va]))
    g[va] = np.mean(ps, 0)
    np.savez(_repo_io(out), ids=np.array(ids), y=y, fold=fold, g=g, feature_file=str(fpath), arm=arm)
    print(arm, fs, variant, 'fold', k, 'seconds', round(time.time() - t0, 1), flush=True)
