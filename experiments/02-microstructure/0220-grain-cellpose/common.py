# v1 (superseded by models_v2.py; data-dependent POSITIVE columns). Kept for the record of evaluate/ablation runs.
"""Shared KRR factory (identical to the one in evaluate.py / robustness.py)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold

R = Path(__file__).resolve().parent
_F = np.load(_repo_path(R, 'features-train.npz'))
keys = list(_F['keys'])
POSITIVE = [j for j in range(len(keys)) if np.nanmin(_F['X'][:, j]) >= 0]


def logpos(A):
    A = A.copy(); A[:, POSITIVE] = np.log1p(A[:, POSITIVE]); return A


def krr():
    return GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), FunctionTransformer(logpos), StandardScaler(), KernelRidge(kernel='rbf')),
                        {'kernelridge__alpha': [0.03, 0.1, 0.3, 1.0], 'kernelridge__gamma': [0.005, 0.01, 0.02, 0.05]},
                        cv=KFold(4, shuffle=True, random_state=0), scoring='neg_mean_squared_error')
