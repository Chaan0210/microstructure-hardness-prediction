"""Combine saved dev OOFs (dev-<tag>.npz from dev_eval.py): grain = mean of the listed tags; report FIXED-weight blends per partition and
pooled RMSE over the 400 ids (mean SE across the 3 partitions). Candidates are compared at the pre-fixed weight W=0.75 (grain).
No data-driven weight choice here: picking w on the OOF rows of the outer-train folds is not nested (those OOF predictions, for both
best021 and the grain model, came from models trained with the outer-validation fold's labels), and best021 cannot be refit inside folds.
usage: dev_combine.py TAG [TAG ...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np
tags = sys.argv[1:]
D = [np.load(_repo_io(f'dev-{t}.npz')) for t in tags]
parts = ['original', '20261031', '20261101']
ref = D[0]['original_ids']
W = [0.5, 0.6, 0.7, 0.75, 0.8, 1.0]
se = {w: [] for w in W}; out = {}
for n in parts:
    ids = D[0][f'{n}_ids']; y = D[0][f'{n}_y']; p = D[0][f'{n}_p021']; fold = D[0][f'{n}_fold']; f = D[0][f'{n}_failed']
    for d in D[1:]: assert (d[f'{n}_ids'] == ids).all()
    g = np.mean([d[f'{n}_g'] for d in D], 0); g = np.where(f, p, g)
    ix = {i: k for k, i in enumerate(ids)}; j = np.array([ix[i] for i in ref])
    rm = lambda v: float(np.sqrt(np.mean((v - y) ** 2)))
    res = {'grain': rm(g)}
    for w in W:
        b = (1 - w) * p + w * g; res[f'w{w}'] = rm(b); se[w].append(((b - y) ** 2)[j])
    out[n] = res
    print(n, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}))
pooled = {f'w{w}': float(np.sqrt(np.mean(se[w], 0).mean())) for w in W}
print('pooled', json.dumps({k: round(v, 4) for k, v in pooled.items()}))
