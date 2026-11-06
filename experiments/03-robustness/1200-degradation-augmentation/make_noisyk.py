"""Candidate = 053 on rows with pixel noise < 12 (and failed rows); on test rows with noise >= 12 the partner share of the two sim-augmented linear
models (mean of RidgeCV and log-Ridge alpha 2, each fitted on 500 train rows + degraded copies deg2,deg3 of the clean train images) is raised
from 0.256 to min(0.256*K, 1), the rest of 053 (021, grain block, set, gmm) scaled proportionally.
Dev (053 structure, same gate): K=1 9.6861, K=2 9.5819, K=2.5 9.5480, K=3 9.5263 (053 9.7665); honest half-split fitted K median 3.7.
usage: make_noisyk.py CID K"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, json, hashlib, numpy as np
from pathlib import Path
cid, K = sys.argv[1], float(sys.argv[2])
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, f'submission-candidate-{cid}.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
c53 = rd('053'); N = np.load(_repo_io('fin-none.npz')); G = np.load(_repo_io('fin-d23.npz')); assert [str(i) for i in G['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); T = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
failed = R['failed'] | T['failed']; gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~failed
part0 = (N['ridge'] + N['logridge']) / 2; part1 = (G['ridge'] + G['logridge']) / 2
other = (c53 - 0.256 * part0) / 0.744; w = min(0.256 * K, 1.0)
cand = np.where(gate, (1 - w) * other + w * part1, c53)
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    wr = csv.writer(f); wr.writerow(['ID', 'hardness']); [wr.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c53
info = dict(candidate=cid, K=K, partner_share_noisy=w, formula='noise>=12: (1-w)*other053 + w*mean(ridge_aug, logridge_aug); else 053', n_gated=int(gate.sum()),
            change_rms_all=float(np.sqrt((d ** 2).mean())), change_rms_gated=float(np.sqrt((d[gate] ** 2).mean())), change_mean_gated=float(d[gate].mean()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, f'submission-candidate-{cid}.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
