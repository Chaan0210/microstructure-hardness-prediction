"""Grain-set regressor with the image-level inputs (raw-summary skip Ridge features and global_X) scale-aligned per noise band exactly as
push-900/qm_feats.py (reference = train rows below the median production noise, three noisy bands, robust median/IQR maps fitted on the
500 train rows, label-free). Per-grain object features are left as they are. Dev protocol, outputs var-qm/seed<k>/oof.npz.
usage: setqm.py SEED[,SEED...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import copy, importlib.util, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; S9 = _repo_path(HERE.parents[1], 'probabilistic-grains-900'); E = _RepoPath(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('grain_set_run', _repo_path(HERE, 'run.py')); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
ORIG = copy.deepcopy(run.CONFIG); run.torch.set_num_threads(2)
def rs(x):
    x = x[np.isfinite(x)]
    if len(x) < 5: return np.nan, np.nan
    q1, q2, q3 = np.quantile(x, [.25, .5, .75]); return q2, (q3 - q1) / 1.349
def noise_bands(train_ids, ids):
    F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
    nz_tr = F['X'][[pos[i] for i in train_ids], keys.index('noise')].astype(float); fill = np.nanmedian(nz_tr); nz_tr = np.where(np.isnan(nz_tr), fill, nz_tr)
    med = np.median(nz_tr); edges = np.quantile(nz_tr[nz_tr >= med], [1 / 3, 2 / 3])
    return med, edges, fill
def band(nz, med, edges, fill): nz = np.where(np.isnan(nz), fill, nz); return np.where(nz < med, -1, np.digitize(nz, edges))
def align(X, b, Xfit, bfit):
    X = X.copy()
    for c in range(X.shape[1]):
        mr, sr = rs(Xfit[bfit == -1, c])
        for k in range(3):
            mb, sb = rs(Xfit[bfit == k, c])
            if not (np.isfinite(sb) and sb > 1e-12 and np.isfinite(sr)): continue
            m = b == k; X[m, c] = mr + (X[m, c] - mb) * sr / sb
    return X
def aligned_bank(bank, train_bank_ids):
    """train_bank_ids: ids whose rows define the maps (the 500 train rows). Returns the bank with summary/global_X aligned."""
    F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True)
    med, edges, fill = noise_bands([str(i) for i in F['ids']], None)
    return med, edges, fill
if __name__ == '__main__':
    bank, base, matched = run.load_inputs(_repo_path(S9, 'object-bank-train.npz'), _repo_path(S9, 'base.npz'))
    F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
    ids = [str(i) for i in bank['ids']]; med, edges, fill = noise_bands(ids, ids)
    b = band(F['X'][[pos[i] for i in ids], keys.index('noise')].astype(float), med, edges, fill)
    bank['summary'] = align(bank['summary'], b, bank['summary'], b); bank['global_X'] = align(bank['global_X'], b, bank['global_X'], b)
    for seed in [int(s) for s in sys.argv[1].split(',')]:
        run.CONFIG = {**copy.deepcopy(ORIG), 'seed': seed}; out = _repo_path(HERE, 'var-qm', f'seed{seed}'); out.mkdir(parents=True, exist_ok=True)
        run.run_train(bank, base, matched, _repo_path(S9, 'object-bank-train.npz'), _repo_path(S9, 'base.npz'), out, 'cpu'); print('done', seed, flush=True)
