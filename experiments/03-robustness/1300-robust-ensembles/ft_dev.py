"""Fine-tuned TabPFN v3.5 (tabpfn.finetuning.FinetunedTabPFNRegressor) on the Study900 dev folds for one feature file.
usage: ft_dev.py FEAT_TRAIN_NPZ TAG EPOCHS LR PARTS(e.g. 0 or 0,1,2) [FOLDS]  -> ft-<TAG>.npz (OOF [3,400], NaN where not run)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, socket, csv, numpy as np, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
sys.path.insert(0, _repo_io('../push-300')); import harness
def blocked(*a, **k): raise RuntimeError('offline')
socket.create_connection = blocked; socket.getaddrinfo = blocked
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
from tabpfn.constants import ModelVersion
from pathlib import Path
fn, tag, ep, lr = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]); parts = [int(x) for x in sys.argv[5].split(',')]; folds = [int(x) for x in sys.argv[6].split(',')] if len(sys.argv) > 6 else range(5)
P = Path(_repo_io('../..')); lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sid = load(3)['ids']; F = np.load(_repo_io(fn), allow_pickle=True); pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(np.float32)
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
import os; ES = os.environ.get('ES', '1') == '1'
mp = str(harness.W3 / 'tabpfn-v3.5-20260909.safetensors'); O = np.full((3, 400), np.nan)
for p_ in parts:
    d = np.load(_repo_io(splits[p_])); ids, fold = [str(i) for i in d['ids']], d['fold']
    for k in folds:
        tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; t0 = time.time()
        m = FinetunedTabPFNRegressor(device='mps', epochs=ep, learning_rate=lr, random_state=k, extra_regressor_kwargs={'model_path': mp}, model_version=ModelVersion('v3.5'),
                                     n_estimators_final_inference=8, early_stopping=ES, early_stopping_patience=4, save_checkpoint_interval=None, validation_split_ratio=(0.1 if ES else None))
        m.fit(XA[[pos[i] for i in tr]], np.array([lab[i] for i in tr], np.float32)); pr = m.predict(XA[[pos[i] for i in va]])
        O[p_, [sid.index(i) for i in va]] = pr; yv = np.array([lab[i] for i in va])
        print(f'part {p_} fold {k}: RMSE {np.sqrt(((pr - yv) ** 2).mean()):.3f} sec {time.time() - t0:.0f}', flush=True)
        np.savez(_repo_io(f'ft-{tag}.npz'), oof=O)
