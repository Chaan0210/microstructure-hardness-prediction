import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path('../../artifacts'); out = A / 'submission-candidate-051.csv'; assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open('../../data/sample_submission.csv'))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(A / f'submission-candidate-{c}.csv'))])
p = rd('021'); g = np.mean([p + (rd(c) - p) / 0.75 for c in ('038', '039', '044')], 0)
R = np.load('../probabilistic-grains-900/ridge/final500/predictions.npz'); failed = R['failed']; r = R['raw_ridge']
rL = (rd('050') - 0.2 * p - 0.48 * g) / 0.32
cand = np.where(failed, p, 0.2 * p + 0.48 * g + 0.16 * r + 0.16 * rL)
assert np.allclose(np.where(failed, p, cand), np.where(failed, p, rd('048') + 0.5 * (rd('050') - rd('045'))))
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate='051', formula='where(n_obj<20, p021, 0.2*p021 + 0.48*grain045 + 0.16*ridge500 + 0.16*ridgeLog_a2_var4)', dev=9.8190,
            change_rms_vs_050=float(np.sqrt(((cand - rd('050')) ** 2).mean())), CSV_sha256=sha, uploaded=False)
json.dump(info, open(A / 'submission-candidate-051.manifest.json', 'x'), indent=1); print(json.dumps(info, indent=1))
