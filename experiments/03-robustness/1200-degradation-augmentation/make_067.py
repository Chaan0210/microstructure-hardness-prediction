"""Candidate 067 = 061 + 1[noise >= 12 & ~failed] * 0.128 * sum over the three TabPFN slots (038/039/044) of (TabPFN + sim-augmented log-Ridge
column - original slot): the TA slot replacement of 064, but on the Public-best 061 instead of 062 (single-factor test of TA; 064 bundled the
062 partner set, which the blend-level holdout check does not support). Slot TA test predictions from final_ta.py (fin-ta38/fin-ta39/fin-ta);
original slot predictions recovered exactly from the 038/039/044 CSVs. Nothing fitted on test data.
Dev 9.6105 (061 9.6636; partitions +0.066/+0.061/+0.032), K_eff 1.29, holdout blend noisy dMSE vs 061 -2.69 (SE 4.29). usage: make_067.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-067.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p, c61 = rd('021'), rd('061'); G = np.load(_repo_io('fin-d1to7n12-wal.npz'))
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); S = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~(R['failed'] | S['failed'])
delta = np.zeros(1000); info_slots = {}
for c, f in (('038', 'fin-ta38.npz'), ('039', 'fin-ta39.npz'), ('044', 'fin-ta.npz')):
    T = np.load(_repo_io(f)); assert [str(i) for i in T['ids']] == order; g = (rd(c) - 0.25 * p) / 0.75; d = T['g44ta'] - g; delta += 0.128 * d
    info_slots[c] = dict(rms_change_gated=float(np.sqrt((d[gate] ** 2).mean())), mean_change_gated=float(d[gate].mean()))
cand = c61 + gate * delta; assert np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate='067', formula='061 + 1[noise>=12 & ~failed]*0.128*sum_slots(g_TA - g)', dev=9.6105, n_gated=int(gate.sum()), slots=info_slots,
            change_rms_vs_061=float(np.sqrt(((cand - c61) ** 2).mean())), change_rms_vs_064=float(np.sqrt(((cand - rd('064')) ** 2).mean())),
            change_rms_vs_053=float(np.sqrt(((cand - rd('053')) ** 2).mean())), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-067.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
