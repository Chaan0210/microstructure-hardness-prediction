"""Candidate 053 = 0.80*051 + 0.05*grainSet + 0.15*gmmRidge: two further partners on top of 051 (dev 9.8190 -> 9.7665, partitions +0.058/+0.070/+0.029).
grainSet = Study900 grain-set regressor (3 seeds, 500 train rows; set_model/finalize500.py); gmmRidge = Study900 image_gmm__x98_augmented Ridge
(98 production columns + per-image GMM phase measurements of the zero-shot grain objects), refit on the 500 train rows by gmm_partner.py final
(dev OOF reproduced exactly). Failed images keep best021.
usage (from experiments/push-900): python make_053.py   -> ../../artifacts/submission-candidate-053.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-053.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p = rd('021'); c51 = rd('051')
T = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True); assert [str(i) for i in T['ids']] == order
Gm = np.load(_repo_io('gmm-partner-test.npz'), allow_pickle=True); assert [str(i) for i in Gm['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); failed = R['failed'] | T['failed']
cand = np.where(failed, p, 0.80 * c51 + 0.05 * T['set_mean_raw'] + 0.15 * Gm['pred'])
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate='053', formula='where(failed, p021, 0.80*cand051 + 0.05*grainSet3seed + 0.15*gmmRidge500)', dev_051=9.8190, dev_053=9.7665,
            gmm_vs_051_rms=float(np.sqrt(np.mean(np.where(failed, 0, Gm['pred'] - c51) ** 2))), change_rms_vs_051=float(np.sqrt(((cand - c51) ** 2).mean())),
            n_failed=int(failed.sum()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-053.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
