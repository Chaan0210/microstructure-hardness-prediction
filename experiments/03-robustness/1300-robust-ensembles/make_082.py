"""Candidates 082 = 076 + residual-CNN correction (round4 final500 'addition' minus 073; 318 noise>=12 rows), and 083 = 082 + (079 - 078)
(the sigma-2/4 cumulant TabPFN-member change of 079, without 078's partner change). Nothing fitted on test.
Public arithmetic (076 8.9647, 078 9.0299, 080 8.9998; the 078 partner change and the correction act on disjoint rows): 082 reads 8.9343.
usage (from experiments/push-1300): python make_082.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); R4 = Path(_repo_io('../rmse-research-20261015/round4-full073/final500'))
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda p: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_io(p)))])
c = {k: rd(_repo_path(A, f'submission-candidate-{k}.csv')) for k in ('073', '076', '078', '079')}
corr = rd(_repo_path(R4, 'candidate-aux-noise-addition.csv')) - c['073']
for cid, cand, formula in (('082', c['076'] + corr, '076 + (round4 addition - 073)'), ('083', c['076'] + corr + (c['079'] - c['078']), '076 + (round4 addition - 073) + (079 - 078)')):
    out = _repo_path(A, f'submission-candidate-{cid}.csv'); assert not out.exists() and np.isfinite(cand).all() and len(cand) == 1000
    with out.open('x', newline='') as f:
        w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
    sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c['076']
    info = dict(candidate=cid, formula=formula, change_rms_vs_076=float(np.sqrt((d ** 2).mean())), CSV_sha256=sha, uploaded=False)
    json.dump(info, open(_repo_path(A, f'submission-candidate-{cid}.manifest.json'), 'x'), indent=1); print(json.dumps(info))
