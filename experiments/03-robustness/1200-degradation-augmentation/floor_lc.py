"""Learning curve of TabPFN on degraded (deg2) vs clean features, same 249 clean train images: train on n images, evaluate on held-out images
(random splits, 20 repeats with a fixed 49-image evaluation set per repeat)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); pos = {str(i): k for k, i in enumerate(F['ids'])}
S = np.load(_repo_io('../push-500/synthrows-train_deg2-u1dmavgH2Z.npz'), allow_pickle=True); sid = [str(i) for i in S['ids']]
y = np.array([lab[i] for i in sid]); XA = F['X'][[pos[i] for i in sid]].astype(float); XB = S['X'].astype(float)
dev_ = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
res = {}
for rep in range(12):
    r = np.random.RandomState(rep); p = r.permutation(len(y)); ev, pool = p[:49], p[49:]
    for n in (100, 150, 200):
        tr = pool[:n]
        for nm, X in (('clean', XA), ('deg2', XB)):
            m = harness.tabpfn_new('v3.5', rep, dev_); m.fit(X[tr], y[tr]); res.setdefault((nm, n), []).append(((m.predict(X[ev]) - y[ev]) ** 2).mean())
    print(rep, {f'{k[0]}{k[1]}': round(float(np.sqrt(np.mean(v))), 3) for k, v in res.items()}, flush=True)
