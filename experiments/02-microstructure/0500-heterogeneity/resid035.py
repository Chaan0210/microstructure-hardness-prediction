"""Shared helper: 035-configuration dev OOF (mean of 4 TabPFN-3.5 variant OOFs, fixed W=0.75 with best021) averaged over the 3 partitions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
from pathlib import Path
H = Path(__file__).resolve().parent
P3 = _repo_path(H.parent, 'push-300')
TAGS = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
PARTS = ['original', '20261031', '20261101']

def load(tags=TAGS, w=0.75):
    D = [np.load(_repo_path(P3, f'dev-{t}-v35.npz')) for t in tags]
    ref = sorted(str(i) for i in D[0]['original_ids'])
    B, G, P = [], [], []
    for n in PARTS:
        I = [str(i) for i in D[0][n + '_ids']]; ix = {i: k for k, i in enumerate(I)}; j = [ix[i] for i in ref]
        p = D[0][n + '_p021']; f = D[0][n + '_failed']
        g = np.mean([d[n + '_g'] for d in D], 0); g = np.where(f, p, g)
        B.append(((1 - w) * p + w * g)[j]); G.append(g[j]); P.append(p[j])
        y = D[0][n + '_y'][j]
    return np.array(ref), y, np.array(B), np.array(G), np.array(P)

def noise():
    ids = [r['ID'] for r in csv.DictReader(open(_repo_path(_RepoPath(__file__).resolve().parents[3], 'data/train.csv')))]; nz = np.load(_repo_path(P3, 'noise_train.npy'))
    return dict(zip(ids, nz))

def labels():
    return {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(_RepoPath(__file__).resolve().parents[3], 'data/train.csv')))}
