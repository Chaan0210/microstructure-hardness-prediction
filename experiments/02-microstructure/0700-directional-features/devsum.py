"""Pooled fixed-W dev summary (same pooling as push-300/dev_eval.py se()), for one or more dev npz files (grain OOF averaged across files)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
parts = ['original', '20261031', '20261101']
def pooled(files, w=0.75):
    D = [np.load(_repo_io(f), allow_pickle=True) for f in files]; ref = [str(i) for i in D[0]['original_ids']]; out = []; per = []
    for n in parts:
        ids = [str(i) for i in D[0][f'{n}_ids']]; ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]
        p, f, y = D[0][f'{n}_p021'][j], D[0][f'{n}_failed'][j], D[0][f'{n}_y'][j]; g = np.mean([d[f'{n}_g'] for d in D], 0)[j]
        se = ((1 - w) * p + w * np.where(f, p, g) - y) ** 2; out.append(se); per.append(np.sqrt(se.mean()))
    return float(np.sqrt(np.mean(out, 0).mean())), per
if __name__ == '__main__':
    w = float(sys.argv[1]); groups = sys.argv[2:]
    for g in groups:
        files = g.split('+'); r, per = pooled(files, w); print(f'{r:.4f}  per-part {" ".join(f"{v:.3f}" for v in per)}  {g}')
