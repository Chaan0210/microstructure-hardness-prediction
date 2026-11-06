"""Candidate 052 = 0.92*051 + 0.08*grainSet: Study900's grain-set regressor (small set network over per-grain objects of the zero-shot masks + Ridge skip,
3 seeds, fitted on the 500 train rows: probabilistic-grains-900/set_model/finalize500.py) as a third, very different partner (error corr 0.77 with 051).
Dev on top of 051: share 0.08 -> 9.8190 -> 9.7816 (+0.037, partitions +0.043/+0.069/+0.000, CI [-0.03, +0.11]). Failed images keep best021.
usage (from experiments/push-900): python make_052.py   -> ../../artifacts/submission-candidate-052.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-052.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p = rd('021'); c51 = rd('051')
T = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True); assert [str(i) for i in T['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); failed = R['failed'] | T['failed']
s = T['set_mean_raw']; cand = np.where(failed, p, 0.92 * c51 + 0.08 * s)
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate='052', formula='where(failed, p021, 0.92*cand051 + 0.08*grainSet3seed)', dev_051=9.8190, dev_052=9.7816, set_vs_051_rms=float(np.sqrt(np.mean(np.where(failed, 0, s - c51) ** 2))),
            change_rms_vs_051=float(np.sqrt(((cand - c51) ** 2).mean())), n_failed=int(failed.sum()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-052.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
