"""Candidate 046 = 0.20*best021 + 0.80*mean(grain of 038, 039, 044, 046c12, 046c18, 046zs, 046zsp).
Component CSVs are 0.25*best021 + 0.75*grain (best021 where n_obj < 20) -> grain recovered exactly. 046c12/046c18: 039 + shading-corrected
phase fraction at 12/18 gray; 046zs: 044 with z_f25,z_f50 from the shading-target U-Net z1s; 046zsp: 046zs + z1s m_ph, m_ph_hard.
usage (from experiments/push-800): python make_046.py [OUT_CSV]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = Path(sys.argv[1]) if len(sys.argv) > 1 else _repo_path(A, 'submission-candidate-046.csv')
assert not out.exists(), 'refuse overwrite'
rd = lambda c: {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))}
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
p = np.array([rd('021')[i] for i in order]); comps = ('038', '039', '044', '046c12', '046c18', '046zs', '046zsp')
g = np.mean([p + (np.array([rd(c)[i] for i in order]) - p) / 0.75 for c in comps], 0)
cand = p + 0.80 * (g - p); assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
c45 = np.array([rd('045')[i] for i in order])
print(out, hashlib.sha256(out.read_bytes()).hexdigest(), 'rms change vs 045 %.3f' % np.sqrt(np.mean((cand - c45) ** 2)))
