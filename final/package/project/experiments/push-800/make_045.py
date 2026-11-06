"""Candidate 045 = 0.20*best021 + 0.80*mean(grain_038, grain_039, grain_044).
Each candidate c in {038, 039, 044} is 0.25*best021 + 0.75*grain_c (best021 where n_obj < 20), so grain_c = best021 + (cand_c - best021)/0.75 exactly.
usage (from experiments/push-800): python make_045.py [OUT_CSV]   (default ../../artifacts/submission-candidate-045-repro.csv; never overwrites)"""
import csv, sys, hashlib, numpy as np
from pathlib import Path
A = Path('../../artifacts'); out = Path(sys.argv[1]) if len(sys.argv) > 1 else A / 'submission-candidate-045-repro.csv'
assert not out.exists(), 'refuse overwrite'
rd = lambda c: {r['ID']: float(r['hardness']) for r in csv.DictReader(open(A / f'submission-candidate-{c}.csv'))}
order = [r['ID'] for r in csv.DictReader(open('../../data/sample_submission.csv'))]
p = np.array([rd('021')[i] for i in order])
g = np.mean([p + (np.array([rd(c)[i] for i in order]) - p) / 0.75 for c in ('038', '039', '044')], 0)
cand = p + 0.80 * (g - p)
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
print(out, sha, 'matches submitted 045:', sha == '7a9dbb257f9c599596769cfcade50f8dd3d988637879ff9896113c62138725c1')
