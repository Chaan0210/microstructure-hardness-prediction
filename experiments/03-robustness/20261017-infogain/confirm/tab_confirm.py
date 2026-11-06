"""H3 confirmation runner: one TabPFN block member (slot x variant x feature set) with or without +im_mean, production dev protocol
(push-300/dev_eval.py logic) on one partition, or the 400->100 holdout fit.
usage: tab_confirm.py ARM SLOT VARIANT FS PARTITION      ARM base|mean; PARTITION original|20261031|20261101|holdout
Writes confirm/tab/{ARM}-{FS}-{SLOT}-{VARIANT}-{PARTITION}.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
import sys
sys.dont_write_bytecode = True
import time
import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
E = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_path(E, 'push-300')))
import harness  # noqa: E402

arm, slot, variant, fs, part = sys.argv[1:6]
device = os.environ.get('TAB_DEVICE', 'cpu')
tag = f'h9phys-{fs}-044-{variant}' if slot == '044' else f'h9-{fs}-{slot}-{variant}'
fpath = _repo_path(E, f'push-1300/abl/feat-{tag}-train.npz')
F = np.load(_repo_io(fpath), allow_pickle=True); pos = {str(i): k for k, i in enumerate(F['ids'])}
XA = F['X'].astype(float)
S = np.load(_repo_path(E, 'infogain-20261017/h3/spec-train.npz'), allow_pickle=True)
assert [str(i) for i in S['ids']] == [str(i) for i in F['ids']]
if arm == 'mean':
    XA = np.c_[XA, S['X'][:, 0]]
else:
    assert arm == 'base'
out = _repo_path(HERE, f'tab/{arm}-{fs}-{slot}-{variant}-{part}.npz')
out.parent.mkdir(exist_ok=True)
t0 = time.time()
if part == 'holdout':
    inp = np.load(_repo_path(E, 'rmse-research-20261015/inputs.npz'), allow_pickle=True)
    ids_all = [str(i) for i in inp['ids']]; dev = set(inp['dev'].tolist())
    tr_ids = [ids_all[i] for i in sorted(dev)]; ho_ids = [ids_all[i] for i in range(500) if i not in dev]
    ytr = inp['y'][sorted(dev)]
    Xtr = XA[[pos[i] for i in tr_ids]]; Xho = XA[[pos[i] for i in ho_ids]]
    ps = []
    for sd in (0, 1, 2):
        m = harness.tabpfn_new('v3.5', 100 * sd, device, None); m.fit(Xtr, ytr); ps.append(m.predict(Xho))
    np.savez(_repo_io(out), ids=np.array(ho_ids), g=np.mean(ps, 0), y=inp['y'][[i for i in range(500) if i not in dev]], feature_file=str(fpath))
else:
    path = {'original': _repo_path(E, 'resolution-216/ensemble-OOF.npz'),
            '20261031': _repo_path(E, 'resolution-repartition-217/20261031/ensemble-OOF.npz'),
            '20261101': _repo_path(E, 'resolution-repartition-217/20261101/ensemble-OOF.npz')}[part]
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']
    X = XA[[pos[i] for i in ids]]
    g = np.full(len(y), np.nan)
    for k in range(5):
        tr, va = fold != k, fold == k; ps = []
        for sd in (0, 1, 2):
            m = harness.tabpfn_new('v3.5', 100 * sd + k, device, None); m.fit(X[tr], y[tr]); ps.append(m.predict(X[va]))
        g[va] = np.mean(ps, 0)
    np.savez(_repo_io(out), ids=np.array(ids), y=y, fold=fold, g=g, feature_file=str(fpath))
print(arm, slot, variant, fs, part, 'seconds', round(time.time() - t0, 1), flush=True)
