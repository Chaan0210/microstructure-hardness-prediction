"""Candidate 063 = 062 + on test rows with pixel noise >= 12: the 044 TabPFN slot (0.128 of 053 = 0.384/3) replaced by the same slot refitted with
an extra column = sim-augmented log-Ridge prediction (final_ta.py -> fin-ta.npz; train column 5-fold OOF, test column full fit).
Original 044-slot test predictions recovered exactly from the submitted 044 CSV (0.25*021 + 0.75*grain). Dev (053 structure, same gates):
053 9.7665, 062 9.6448, 063 9.6316 (partitions 9.6252/9.6283/9.6413). usage: make_063.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-063.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p, c62 = rd('021'), rd('062'); g44 = (rd('044') - 0.25 * p) / 0.75
T = np.load(_repo_io('fin-ta.npz')); assert [str(i) for i in T['ids']] == order; G = np.load(_repo_io('fin-d1to7n12-wal.npz'))
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); S = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~(R['failed'] | S['failed'])
cand = c62 + gate * 0.128 * (T['g44ta'] - g44); assert np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c62
info = dict(candidate='063', formula='062 + 1[noise>=12 & ~failed]*0.128*(g44_TA - g44)', dev=9.6316, n_gated=int(gate.sum()),
            g44ta_vs_g44_rms_gated=float(np.sqrt(((T['g44ta'] - g44)[gate] ** 2).mean())), g44ta_vs_g44_rms_clean=float(np.sqrt(((T['g44ta'] - g44)[~gate] ** 2).mean())),
            change_rms_vs_062=float(np.sqrt((d ** 2).mean())), change_rms_vs_053=float(np.sqrt(((cand - rd('053')) ** 2).mean())), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-063.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
