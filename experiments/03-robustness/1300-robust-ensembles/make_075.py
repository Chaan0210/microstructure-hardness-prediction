"""Candidate 075 = equal-weight average of 061 and 073 (pre-specified hedge; no coefficient fitted).
Identity on any row set (Public, Private): MSE((a+b)/2) = (MSE(a)+MSE(b))/2 - mean((a-b)^2)/4; here mean((a-b)^2)/4 = 0.156 on all 1000 test rows,
so the average beats both members whenever their MSEs differ by less than 0.31 (about 0.017 RMSE). Train evidence favours 073 (dev +0.079), Public
favours 061 (+0.0033 = tie within SE 0.04): the equal average is the minimum-expected-loss choice under that split posterior.
usage (from experiments/push-1300): python make_075.py -> ../../artifacts/submission-candidate-075.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-075.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
a, b = rd('061'), rd('073'); cand = 0.5 * (a + b); assert np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate='075', formula='0.5*061 + 0.5*073', spread_term=float(((a - b) ** 2).mean() / 4), predicted_public_from_identity=float(np.sqrt(0.5 * (8.9831248984 ** 2 + 8.9863906091 ** 2) - ((a - b) ** 2).mean() / 4)),
            CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-075.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
