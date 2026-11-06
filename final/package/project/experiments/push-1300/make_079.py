"""Candidate 079 = 078 + sigma-2/4 cumulant p34 and contrast (4 more label-blind columns) in all 12 TabPFN block members (nb+h9+4 on non-gated rows,
full+h9+4 on gated rows). Same weights. Nothing fitted on test. usage (from experiments/push-1300): python make_079.py"""
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path('../../artifacts'); out = A / 'submission-candidate-079.csv'; assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open('../../data/sample_submission.csv'))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(A / f'submission-candidate-{c}.csv'))])
c78 = rd('078'); G = np.load('../push-1200/fin-d1234n12.npz'); R = np.load('../probabilistic-grains-900/ridge/final500/predictions.npz'); S = np.load('../probabilistic-grains-900/set_model/full500/predictions.npz', allow_pickle=True)
failed = R['failed'] | S['failed']; ng = (np.nan_to_num(G['noise'], nan=0) < 12) & ~failed
T0 = np.load('fin-h9-tab-cpu.npz'); T1 = np.load('fin-h9s2-tab-cpu.npz'); assert [str(i) for i in T1['ids']] == order
blk = lambda T: np.where(ng, np.mean([T[f'nb_{c}'] for c in ('038', '039', '044')], 0), np.mean([T[f'full_{c}'] for c in ('038', '039', '044')], 0))
cand = np.where(failed, c78, c78 + 0.384 * (blk(T1) - blk(T0))); assert np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c78
info = dict(candidate='079', formula='078 + 0.384*(TabPFN block with sigma-2/4 cumulant cols - block)', change_rms_vs_078=float(np.sqrt((d ** 2).mean())), change_rms_gated=float(np.sqrt((d[~ng & ~failed] ** 2).mean())), CSV_sha256=sha, uploaded=False)
json.dump(info, open(A / 'submission-candidate-079.manifest.json', 'x'), indent=1); print(json.dumps(info, indent=1))
