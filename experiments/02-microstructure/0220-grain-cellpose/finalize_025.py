"""Candidate 025 = 0.5*best021 + 0.5*TabPFN-v2(v3 grain features), arm C of plan-v4-tabpfn.json. Offline, existing private weights.
LICENSE NOTE: TabPFN v2 weights are under Prior Labs License 1.1 (Apache-2.0 derivative + attribution); use in a submission is the user's decision."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, sys, socket, warnings
import numpy as np, torch
from pathlib import Path
warnings.filterwarnings('ignore')
def blocked(*a, **k): raise RuntimeError('offline')
socket.create_connection = blocked; socket.socket.connect = blocked; socket.getaddrinfo = blocked
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]; T = _repo_path(P, 'experiments/tabpfn-v2'); sys.path.insert(0, str(R))
from finalize_common import load_train_test, write_candidate, sha
from tabpfn import TabPFNRegressor, ModelSpecs
from tabpfn import model_loading
ex = json.load(open(_repo_path(R, 'explore-tabpfn.json'))); assert ex['bC_pooled']['all3_better'] and ex['bC_pooled']['gain'] >= 0.10
models, criterion, configs, ic = model_loading.load_model_criterion_config(model_path=_repo_path(T, 'weights/tabpfn-v2-regressor.ckpt'), version='v2', estimator_type='regressor', check_bar_distribution_criterion=True, download_if_not_exists=False)
spec = ModelSpecs(model=models[0], architecture_config=configs[0], inference_config=ic, norm_criterion=criterion)
torch.set_num_threads(4)
Xtr, ytr, Xte, tids, keys = load_train_test()
est = TabPFNRegressor(model_path=spec, n_estimators=8, device='cpu', inference_precision=torch.float32, random_state=0, show_progress_bar=False)
est.fit(Xtr, ytr); g = est.predict(Xte)
write_candidate('025', g, tids, keys, Xte, 0.5, 20,
                dict(formula='0.5*best021 + 0.5*TabPFN-v2(v3 grain features) (fallback best021 if n_obj<20)', tabpfn_weights_sha256=sha(_repo_path(T, 'weights/tabpfn-v2-regressor.ckpt')),
                     tabpfn_license='Prior Labs License 1.1 (Apache-2.0 derivative + attribution clause)', n_estimators=8, random_state=0,
                     dev_evidence='explore-tabpfn.json arm C: pooled 3-partition OOF 11.750 vs v3 blend 11.872 (gain 0.121, CI [-0.024,0.263]); all 3 partitions better.'))
