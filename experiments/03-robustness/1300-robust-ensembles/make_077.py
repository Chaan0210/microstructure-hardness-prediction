"""Candidate 077 = 076 + the two H9 cumulant columns also in the plain linear partners (RidgeCV, logRidge x4, gmm-Ridge) used on non-gated rows (noise < 12).
Train-only evidence: dev +0.0173 over 076 (+0.007/+0.030/+0.015) vs random-2-column control +0.0044; holdout-100 partners all improve
(ridge 9.059->9.016, logRidge 9.537->9.413, gmm 8.693->8.668), proxy blend dMSE -0.86 +- 0.86. Nothing fitted on test.
usage (from experiments/push-1300): python make_077.py -> ../../artifacts/submission-candidate-077.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-077.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
c76 = rd('076'); G = np.load(_repo_io('../push-1200/fin-d1234n12.npz')); N = np.load(_repo_io('fin-nb-partners.npz')); H = np.load(_repo_io('fin-h9-partners.npz')); assert [str(i) for i in H['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); S = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
failed = R['failed'] | S['failed']; ng = (np.nan_to_num(G['noise'], nan=0) < 12) & ~failed
delta = 0.128 * (H['ridge'] - N['ridge_nb']) + 0.128 * (H['logridge'] - N['logridge_nb']) + 0.15 * (H['gmm'] - N['gmm_nb'])
cand = np.where(ng, c76 + delta, c76); assert np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c76
info = dict(candidate='077', formula='076 + H9 cumulant columns in plain partners on noise<12 rows', change_rms_vs_076=float(np.sqrt((d ** 2).mean())), change_rms_nongated=float(np.sqrt((d[ng] ** 2).mean())), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-077.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
