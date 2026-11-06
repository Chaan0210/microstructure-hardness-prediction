"""Candidate 061 = 057 with the augmentation set enlarged to every label-preserving degraded copy of the clean train images whose measured pixel
noise is >= 12 gray (deg, deg2, deg3 filtered + deg4 = realdeg2 calibrated noisy preset with target noise from the train >= 12 pool):
final_aug.py with NOISEMIN=12 deg,deg2,deg3,deg4 -> fin-d1234n12. Same gate (test noise >= 12) and weights (0.128 each) as 057.
Dev (053 structure): 9.7665 -> 9.6636, partitions +0.112/+0.089/+0.107 (057: 9.6861); shuffled-label deg4 control -0.095.
usage (from experiments/push-1200): python make_061.py -> ../../artifacts/submission-candidate-061.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-061.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
c53 = rd('053'); N = np.load(_repo_io('fin-none.npz')); G = np.load(_repo_io('fin-d1234n12.npz')); assert [str(i) for i in G['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); T = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
failed = R['failed'] | T['failed']; gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~failed
cand = c53 + gate * (0.128 * (G['ridge'] - N['ridge']) + 0.128 * (G['logridge'] - N['logridge']))
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c53
info = dict(candidate='061', formula='053 + 1[noise>=12 & ~failed]*0.128*((ridge_aug - ridge) + (logridge_aug - logridge)), aug = + all degraded copies (deg, deg2, deg3, deg4) with measured noise >= 12',
            dev_053=9.7665, dev_061=9.6636, n_gated=int(gate.sum()), change_rms_all=float(np.sqrt((d ** 2).mean())), change_rms_gated=float(np.sqrt((d[gate] ** 2).mean())),
            change_mean_gated=float(d[gate].mean()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-061.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
