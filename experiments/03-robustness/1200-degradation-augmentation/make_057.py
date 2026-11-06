"""Candidate 057 = 053 with the Ridge and log-Ridge components (each 0.128 of 053 = 0.8*0.16) replaced on test rows with pixel noise >= 12 gray
by the same models fitted on the 500 train rows PLUS label-preserving degraded copies of the clean train images (push-400/synth/train_deg2,
train_deg3; label-free degradation calibrated on our train images; rows built by synth_anghcor.py; final_aug.py deg2,deg3 fin-d23).
Dev (053 structure, same gate): 9.7665 -> 9.6861, partitions +0.088/+0.063/+0.090; shuffled-label sim control -0.29.
usage (from experiments/push-1200): python make_057.py -> ../../artifacts/submission-candidate-057.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-057.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
c53 = rd('053'); N = np.load(_repo_io('fin-none.npz')); G = np.load(_repo_io('fin-d23.npz')); assert [str(i) for i in G['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); T = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
failed = R['failed'] | T['failed']; gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~failed
cand = c53 + gate * (0.128 * (G['ridge'] - N['ridge']) + 0.128 * (G['logridge'] - N['logridge']))
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c53
info = dict(candidate='057', formula='053 + 1[noise>=12 & ~failed]*0.128*((ridge_aug - ridge) + (logridge_aug - logridge)), aug = + deg2,deg3 copies of clean train images',
            dev_053=9.7665, dev_057=9.6861, n_gated=int(gate.sum()), change_rms_all=float(np.sqrt((d ** 2).mean())), change_rms_gated=float(np.sqrt((d[gate] ** 2).mean())),
            change_mean_gated=float(d[gate].mean()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-057.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
