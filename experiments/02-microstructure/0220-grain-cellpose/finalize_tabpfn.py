"""Build a TabPFN-S candidate: 0.5*best021 + 0.5*mean_{seed in 0,1,2} TabPFN-v2(n_estimators=32) on a given feature set.
Usage (inside experiments/tabpfn-v2 with its .venv): python finalize_tabpfn.py <candidate_id> <feature_prefix> <evidence_json> <evidence_key>
feature_prefix: 'features-v3' or 'features-v3tta' (files <prefix>-train.npz / <prefix>-test.npz)."""
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
from finalize_common import write_candidate, sha
import csv
from tabpfn import TabPFNRegressor, ModelSpecs
from tabpfn import model_loading
cid, prefix, evidence, ekey = sys.argv[1:5]
ev = json.load(open(_repo_path(R, evidence))); assert ev[ekey], f'{evidence}:{ekey} not adopted'
models, criterion, configs, ic = model_loading.load_model_criterion_config(model_path=_repo_path(T, 'weights/tabpfn-v2-regressor.ckpt'), version='v2', estimator_type='regressor', check_bar_distribution_criterion=True, download_if_not_exists=False)
spec = ModelSpecs(model=models[0], architecture_config=configs[0], inference_config=ic, norm_criterion=criterion); torch.set_num_threads(4)
Ftr = np.load(_repo_path(R, f'{prefix}-train.npz')); Fte = np.load(_repo_path(R, f'{prefix}-test.npz')); keys = list(Ftr['keys']); assert keys == list(Fte['keys'])
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
Xtr, ytr, Xte = Ftr['X'], np.array([lab[i] for i in Ftr['ids']]), Fte['X']
preds = []
for sd in (0, 1, 2):
    est = TabPFNRegressor(model_path=spec, n_estimators=32, device='cpu', inference_precision=torch.float32, random_state=100 * sd, show_progress_bar=False)
    est.fit(Xtr, ytr); preds.append(est.predict(Xte))
g = np.mean(preds, 0)
write_candidate(cid, g, list(Fte['ids']), keys, Xte, 0.5, 20,
                dict(formula=f'0.5*best021 + 0.5*mean3seeds TabPFN-v2(n_est=32) on {prefix} (fallback best021 if n_obj<20)', feature_prefix=prefix,
                     tabpfn_weights_sha256=sha(_repo_path(T, 'weights/tabpfn-v2-regressor.ckpt')), tabpfn_license='Prior Labs License 1.1 (Apache-2.0 derivative + attribution clause)',
                     seed_pred_corr=[float(np.corrcoef(preds[0], preds[k])[0, 1]) for k in (1, 2)], dev_evidence=f'{evidence}:{ekey}'))
