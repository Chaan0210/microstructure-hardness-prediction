"""Candidate 065 = 064 with the slot-replacement gate widened to pixel noise >= 9.5 (mid rows are partly structurally degraded; partners keep the >= 12 gate): ALL three TabPFN grain slots (038, 039, 044; each 0.128 of 053) replaced by the same
slots refitted with an extra column = sim-augmented log-Ridge prediction (final_ta.py -> fin-ta38/fin-ta39/fin-ta; train column 5-fold OOF,
test column full fit; augmentation = label-preserving degraded copies of train images with measured noise >= 12). Original slot test predictions
are recovered exactly from the submitted 038/039/044 CSVs (0.25*021 + 0.75*grain). No quantity is fitted on test data.
Dev (053 structure, same gates): 053 9.7665, 061 9.6636, 062 9.6448, 064 9.5951 (partitions 9.5758/9.5875/9.6220). usage: make_064.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-065.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p, c62 = rd('021'), rd('062'); G = np.load(_repo_io('fin-d1to7n12-wal.npz'))
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); S = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~(R['failed'] | S['failed'])
delta = np.zeros(1000); info_slots = {}
for c, f in (('038', 'fin-ta38.npz'), ('039', 'fin-ta39.npz'), ('044', 'fin-ta.npz')):
    T = np.load(_repo_io(f)); assert [str(i) for i in T['ids']] == order; g = (rd(c) - 0.25 * p) / 0.75; d = T['g44ta'] - g; delta += 0.128 * d
    info_slots[c] = dict(rms_change_gated=float(np.sqrt((d[gate] ** 2).mean())), mean_change_gated=float(d[gate].mean()))
gate95 = (np.nan_to_num(G['noise'], nan=0) >= 9.5) & ~(R['failed'] | S['failed']); cand = c62 + gate95 * delta; assert np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate='065', formula='062 + 1[noise>=9.5 & ~failed]*0.128*sum_slots(g_TA - g)', dev=9.5842, n_gated=int(gate.sum()), slots=info_slots,
            change_rms_vs_062=float(np.sqrt(((cand - c62) ** 2).mean())), change_rms_vs_061=float(np.sqrt(((cand - rd('061')) ** 2).mean())),
            change_rms_vs_053=float(np.sqrt(((cand - rd('053')) ** 2).mean())), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-065.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
