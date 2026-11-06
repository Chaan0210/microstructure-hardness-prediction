"""Candidate 024 = 0.5*best021 + 0.5*grain-v3 (family chosen by 5-fold CV on 500, same rule as validate_v3). License-safe (no TabPFN)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, sys, warnings
import numpy as np
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
from pathlib import Path
R = Path(__file__).resolve().parent; sys.path.insert(0, str(R))
from models_v3 import W, MIN_OBJ, select_and_fit
from finalize_common import load_train_test, write_candidate
v = json.load(open(_repo_path(R, 'validation-v3.json'))); assert v['adopted'] == 'B_v3feat_v3mask'
Xtr, ytr, Xte, tids, keys = load_train_test()
f5 = np.zeros(len(ytr), int)
for k, (_, va) in enumerate(KFold(5, shuffle=True, random_state=20261007).split(Xtr)): f5[va] = k
fam, scores, predict, model = select_and_fit(Xtr, ytr, f5, keys)
write_candidate('024', predict(Xte), tids, keys, Xte, W, MIN_OBJ,
                dict(formula='0.5*best021 + 0.5*grain-v3 (fallback best021 if n_obj<20)', family=fam, family_cv_rmse=scores,
                     model_params=(model.best_params_ if hasattr(model, 'best_params_') else None),
                     dev_evidence='validation-v3.json arm B: pooled 3-partition OOF 11.872 vs v2 blend 12.240 (gain 0.368, CI [0.100,0.651]); vs 021 12.815.'))
