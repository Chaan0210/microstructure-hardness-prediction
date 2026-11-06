"""Candidate 047 = 045 with a half-step slope stretch on the low-noise half of the high-noise group (stretch_dev.py: dev 9.9402 -> 9.7105).
Group = features-v3 noise x 255 >= 12; half split at the TRAIN median inside the group (16.31). p' = m + k (p - m), k = 1 + 0.5 (b - 1);
m (mean prediction) and b (OLS slope of y on the 045 dev prediction) come from the dev OOF band rows pooled over the 3 partitions.
Test images only supply their own label-free noise value; nothing is fitted on test.
usage (from experiments/push-800): python make_047.py [OUT_CSV]   (default ../../artifacts/submission-candidate-047.csv; never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = Path(sys.argv[1]) if len(sys.argv) > 1 else _repo_path(A, 'submission-candidate-047.csv')
assert not out.exists(), 'refuse overwrite'
E = '../'; V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; parts = ['original', '20261031', '20261101']
L = lambda fs: [np.load(_repo_io(f), allow_pickle=True) for f in fs]
D38 = L([E + f'push-300/dev-{v}-v35.npz' for v in V]); D39 = L([E + f'push-600/dev-{v}-angh.npz' for v in V]); D44 = L([f'dev-{v}-anghcor.npz' for v in V]); ref = D39[0]
G = lambda Ds, n: np.mean([d[f'{n}_g'] for d in Ds], 0)
nzf = lambda s: (lambda F: ({str(i): x for i, x in zip(F['ids'], F['X'][:, list(F['keys']).index('noise')] * 255)}))(np.load(_repo_io(E + f'grain-cellpose-220/features-v3-{s}.npz'), allow_pickle=True))
ntr, nte = nzf('train'), nzf('test'); vtr = np.array(list(ntr.values())); lo = 12.0; hi = float(np.median(vtr[vtr >= lo]))
Y, P = [], []
for n in parts:
    ids = [str(i) for i in ref[f'{n}_ids']]; q = ref[f'{n}_p021']; f = ref[f'{n}_failed']
    p = 0.2 * q + 0.8 * np.where(f, q, np.mean([G(D38, n), G(D39, n), G(D44, n)], 0)); band = np.array([lo <= ntr[i] < hi for i in ids])
    Y.append(ref[f'{n}_y'][band]); P.append(p[band])
Y, P = np.concatenate(Y), np.concatenate(P); b = float(np.polyfit(P, Y, 1)[0]); m = float(P.mean()); k = 1 + 0.5 * (b - 1)
rd = lambda c: {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))}
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
p45 = np.array([rd('045')[i] for i in order]); tb = np.array([lo <= nte[i] < hi for i in order])
cand = np.where(tb, m + k * (p45 - m), p45)
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as fh:
    w = csv.writer(fh); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate='047', base='045', band=[lo, hi], dev_band_rows=int(len(Y)), slope_b=b, center_m=m, k=k, test_rows_changed=int(tb.sum()),
            change_rms_all=float(np.sqrt(((cand - p45) ** 2).mean())), change_rms_band=float(np.sqrt(((cand - p45)[tb] ** 2).mean())), CSV_sha256=sha)
print(json.dumps(info, indent=1))
if out.parent == A: json.dump(dict(info, uploaded=False), open(_repo_path(A, 'submission-candidate-047.manifest.json'), 'x'), indent=1)
