"""Per-dataset fine-tuned TabPFN on the harness folds. usage: ft_tabpfn.py VERSION EPOCHS LR REPS FOLDS [feat npz]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, socket, numpy as np, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
sys.path.insert(0, _repo_io('../push-300'))
import harness
def blocked(*a, **k): raise RuntimeError('offline')
socket.create_connection = blocked; socket.getaddrinfo = blocked
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
from tabpfn.constants import ModelVersion
ver, epochs, lr = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); reps = [int(x) for x in sys.argv[4].split(',')]; folds = [int(x) for x in sys.argv[5].split(',')]
fn = sys.argv[6] if len(sys.argv) > 6 else '../push-300/feat-a1s250n4vstd-stcoh-train.npz'
F = np.load(_repo_io(fn), allow_pickle=True); X = F['X'].astype(np.float32); ids = [str(i) for i in F['ids']]; y = harness.labels(ids).astype(np.float32); nz = np.load(_repo_io('../push-300/noise_train.npy'))
mp = {'v2': str(harness.T / 'weights/tabpfn-v2-regressor.ckpt'), 'v3.5': str(harness.W3 / 'tabpfn-v3.5-20260909.safetensors')}[ver]
base = np.stack([np.load(_repo_io(f'../push-300/oof_none_v2_feat-a1s250n4vstd-stcoh-train_s{s}.npy')) for s in range(3)]).mean(0)
for r in reps:
    f = harness.folds(len(y), r)
    for j in folds:
        tr, va = f != j, f == j; t0 = time.time()
        m = FinetunedTabPFNRegressor(device='mps', epochs=epochs, learning_rate=lr, random_state=0, extra_regressor_kwargs={'model_path': mp}, model_version=ModelVersion(ver),
                                     n_estimators_final_inference=8, early_stopping=True, early_stopping_patience=4, save_checkpoint_interval=None)
        m.fit(X[tr], y[tr]); p = m.predict(X[va])
        rm = lambda v: float(np.sqrt(np.mean((v - y[va]) ** 2)))
        print(f'rep {r} fold {j}: finetuned {rm(p):.3f}  base(v2 n8 3-seed) {rm(base[r][va]):.3f}  sec {time.time()-t0:.0f}', flush=True)
        np.save(_repo_io(f'ftpfn_{ver}_e{epochs}_r{r}_f{j}.npy'), p)
