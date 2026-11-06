"""Candidate 069 = 061 + 1[noise >= 12 & ~failed] * 0.128 * sum over the three TabPFN slots (038/039/044) of (TA4 slot - original slot).
TA4 = TabPFN-3.5 on the slot features + one column = sim-augmented log-Ridge prediction with the SAME copy set as 061's partners (deg..deg4,
measured noise >= 12; training rows get 5-fold cross-fitted values). Single-factor version of 067 (which used the d1to7 copy set in the column).
Train-only evidence (push-1300): dev plain +0.050, non-stretch +0.027, label-matched noisy dMSE -1.48 (SE 1.32, bootstrap P 0.87), K_eff 1.31;
blend-level holdout-100 noisy dMSE -4.42 (SE 4.63). Original slot predictions recovered exactly from the 038/039/044 CSVs; nothing fitted on test.
usage (from experiments/push-1300): python make_069.py -> ../../artifacts/submission-candidate-069.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-069.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p, c61 = rd('021'), rd('061'); G = np.load(_repo_io('../push-1200/fin-d1234n12.npz')); assert [str(i) for i in G['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); S = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~(R['failed'] | S['failed'])
delta = np.zeros(1000); slots = {}
for c, f in (('038', 'fin-ta4-38.npz'), ('039', 'fin-ta4-39.npz'), ('044', 'fin-ta4-44.npz')):
    T = np.load(_repo_io(f'../push-1200/{f}')); assert [str(i) for i in T['ids']] == order; g = (rd(c) - 0.25 * p) / 0.75; d = T['g44ta'] - g; delta += 0.128 * d
    slots[c] = dict(rms_change_gated=float(np.sqrt((d[gate] ** 2).mean())), mean_change_gated=float(d[gate].mean()))
cand = c61 + gate * delta; assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); c67 = rd('067')
info = dict(candidate='069', formula='061 + 1[noise>=12 & ~failed]*0.128*sum_slots(g_TA4 - g), TA4 column copies = deg,deg2,deg3,deg4 (as 061 partners)', n_gated=int(gate.sum()), slots=slots,
            change_rms_vs_061=float(np.sqrt(((cand - c61) ** 2).mean())), change_rms_vs_067=float(np.sqrt(((cand - c67) ** 2).mean())),
            corr_change_with_067=float(np.corrcoef((cand - c61)[gate], (c67 - c61)[gate])[0, 1]), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-069.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
