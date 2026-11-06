"""Harness: (a) ridge on fine-tuned-encoder embeddings alone; (b) PCA-k of embeddings appended to a feature npz, TabPFN v2 n8.
PCA is fit inside each training fold (no validation rows)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from harness import *
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
emb, fnpz, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
E = np.load(_repo_io(emb), allow_pickle=True); F = np.load(_repo_io(fnpz), allow_pickle=True); ids = [str(i) for i in F['ids']]; assert [str(i) for i in E['ids']] == ids
Z = E['E'].astype(float); X = F['X'].astype(float); y = labels(ids); nz = np.load(_repo_io('noise_train.npy'))
for a in (30, 100, 300, 1000):
    O = cv_predict(lambda: make_pipeline(StandardScaler(), Ridge(alpha=a)), Z, y, reps=(0, 1, 2)); report(f'ridge emb a={a}', O, y, nz)
class Aug:
    def __init__(s): s.m = tabpfn(8, 0, 'cpu'); s.p = make_pipeline(StandardScaler(), PCA(k, random_state=0))
    def fit(s, A, yy): s.p.fit(A[:, X.shape[1]:]); s.m.fit(np.hstack([A[:, :X.shape[1]], s.p.transform(A[:, X.shape[1]:])]), yy); return s
    def predict(s, A): return s.m.predict(np.hstack([A[:, :X.shape[1]], s.p.transform(A[:, X.shape[1]:])]))
O = cv_predict(Aug, np.hstack([X, Z]), y, reps=(0, 1, 2)); report(f'tabpfn {fnpz} + emb PCA{k}', O, y, nz); np.save(_repo_io(f'oof_emb_pca{k}.npy'), O)
