"""Candidate 074 = 073 with the gated-row (noise >= 12 & ~failed) DTA partners refitted with alpha chosen by inner 5-fold CV grouped by source and scored
on real noise >= 12 rows (srcw.py arm B; Ridge keeps alpha 0.1 = production, logRidge alpha 2 -> 0.3 on all 4 variants). Same weights.
Train-only evidence: dev vs 073 +0.0089 (+0.009/+0.003/+0.015), non-stretch +0.006, LM -0.37 (P 0.89); holdout-100 gated partner dMSE -2.57 +- 3.55.
usage (from experiments/push-1300): python make_074.py -> ../../artifacts/submission-candidate-074.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-074.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
c73 = rd('073'); G = np.load(_repo_io('../push-1200/fin-d1234n12.npz')); B = np.load(_repo_io('fin-srcwB.npz')); assert [str(i) for i in B['ids']] == order and [str(i) for i in G['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); S = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~(R['failed'] | S['failed'])
cand = c73 + gate * (0.128 * (B['ridge'] - G['ridge']) + 0.128 * (B['logridge'] - G['logridge'])); assert np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c73
info = dict(candidate='074', formula='073 + gated DTA partners with source-grouped inner-CV alpha (logRidge 2 -> 0.3)', n_gated=int(gate.sum()),
            change_rms_vs_073=float(np.sqrt((d ** 2).mean())), change_rms_gated=float(np.sqrt((d[gate] ** 2).mean())), change_mean_gated=float(d[gate].mean()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-074.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
