"""Candidate 068 = 061 with the raw-image CNN (best021) removed on noisy rows (measured pixel noise >= 12, not failed); its 0.16 share is
redistributed proportionally to the other five components: c068 = (c061 - 0.16 * p021) / 0.84 on those rows, 061 elsewhere.
Evidence (train only): dev-400 x 3 partitions 9.6636 -> 9.5778 (+0.063/+0.081/+0.113), noisy-row dMSE -5.33 (SE 3.37), 11/15 folds;
independent holdout-100 (all components refit on dev-400, TinyCNN stand-in for 021) noisy-row dMSE -5.05 (SE 6.25). Nothing fitted on test.
usage (from experiments/push-1300): python make_068.py -> ../../artifacts/submission-candidate-068.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-068.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p = rd('021'); c61 = rd('061'); G = np.load(_repo_io('../push-1200/fin-d1234n12.npz')); assert [str(i) for i in G['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); T = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
failed = R['failed'] | T['failed']; gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~failed
cand = np.where(gate, (c61 - 0.16 * p) / 0.84, c61)
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c61
info = dict(candidate='068', formula='where(noise>=12 & ~failed, (c061 - 0.16*p021)/0.84, c061)', dev_061=9.6636, dev_068=9.5778,
            n_gated=int(gate.sum()), change_rms_all=float(np.sqrt((d ** 2).mean())), change_rms_gated=float(np.sqrt((d[gate] ** 2).mean())), change_mean_gated=float(d[gate].mean()),
            CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-068.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
