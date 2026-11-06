"""Candidate 071 = 070 with the DTA gate lowered from 12 to 11 (noise 11-12 images carry the noisy rendering: push-1300 render score median -5.14 vs clean -4.2). 070 = 061 with its three Ridge-type partners refitted by heteroscedastic weighted least squares (same weights 0.128/0.128/0.15):
RidgeCV and logRidge (plain on clean/mid rows, DTA deg..deg4 on noise >= 12 rows) and gmm-Ridge, weights 1/sigma^2 with sigma^2 = a + b/N + c*noisy
fitted on 5-fold OOF residuals of the unweighted fit on the 500 train rows (final_wls.py; the unweighted path reproduces 061's components exactly).
Gauss-Markov: WLS with the right variance model is the minimum-variance linear unbiased fit. Evidence: dev +0.017 (parts +0.039/-0.001/+0.013),
noisy non-stretch +0.021, LM -0.70; holdout-100 blend -0.40 MSE (SE 1.23), partners alone better on holdout. Nothing fitted on test."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-071.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
c61 = rd('061'); W0 = np.load(_repo_io('fin-wls0.npz')); W1 = np.load(_repo_io('fin-wls1.npz')); G = np.load(_repo_io('../push-1200/fin-d1234n12.npz')); assert [str(i) for i in W1['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); S = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
failed = R['failed'] | S['failed']; gate = (np.nan_to_num(G['noise'], nan=0) >= 11) & ~failed; gate12 = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~failed
sel = lambda Z, k: np.where(gate, Z[f'{k}_1'], Z[f'{k}_0'])
sel0 = lambda Z, k: np.where(gate12, Z[f'{k}_1'], Z[f'{k}_0'])
delta = 0.128 * (sel(W1, 'ridge') - sel0(W0, 'ridge')) + 0.128 * (sel(W1, 'logridge') - sel0(W0, 'logridge')) + 0.15 * (W1['gmm'] - W0['gmm'])
cand = np.where(failed, c61, c61 + delta); assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c61
info = dict(candidate='071', formula='061 with DTA gate >= 11 and RidgeCV/logRidge/gmm-Ridge partners refitted by WLS (sigma^2 = a + b/N + c*noisy), same weights', dev_061=9.6636, dev_071=9.6439,
            change_rms_all=float(np.sqrt((d ** 2).mean())), change_rms_gated=float(np.sqrt((d[gate] ** 2).mean())), change_rms_clean_mid=float(np.sqrt((d[~gate & ~failed] ** 2).mean())),
            change_mean=float(d.mean()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-071.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
