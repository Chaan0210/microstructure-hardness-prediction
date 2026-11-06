"""K=2 variant (NOT BUILT: gain concentrated in a few rows, bootstrap P 0.65-0.73) = 061 with the DTA partners' share on noisy rows doubled (K = 2: RidgeCV_aug + logRidge_aug 0.256 -> 0.512, the other five
components scaled by 0.488/0.744). Chosen from train-only evidence (push-1300 kcurve.py: non-stretch dev and label-matched dev both peak at
K 1.75-2; blend-level holdout monotone in K). Nothing fitted on test: the gate is each test image's measured noise, K is fixed from train.
usage (from experiments/push-1300): python make_069.py -> ../../artifacts/submission-candidate-k2.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-k2.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
c61 = rd('061'); G = np.load(_repo_io('../push-1200/fin-d1234n12.npz')); assert [str(i) for i in G['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); T = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
failed = R['failed'] | T['failed']; gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~failed
K, aw = 2.0, 0.256; part = (G['ridge'] + G['logridge']) / 2; other = (c61 - aw * part) / (1 - aw)
cand = np.where(gate, (1 - aw * K) * other + aw * K * part, c61)
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c61
info = dict(candidate='069', formula='where(noise>=12 & ~failed, 0.488*other061 + 0.512*(ridge_aug+logridge_aug)/2, c061), other061 = (c061 - 0.256*partners)/0.744',
            dev_061=9.6636, n_gated=int(gate.sum()), change_rms_all=float(np.sqrt((d ** 2).mean())), change_rms_gated=float(np.sqrt((d[gate] ** 2).mean())),
            change_mean_gated=float(d[gate].mean()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-k2.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
