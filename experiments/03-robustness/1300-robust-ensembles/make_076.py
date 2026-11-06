"""Candidate 076 = 074 + H9: all 12 TabPFN block members get the two label-blind cumulant measurements of round2 H9 (sigma-1 p34 phase fraction and
contrast; measure() reproduced exactly on train, re-extracted on test from the same U-Net z1 maps and noise estimate); nb+h9 members on non-gated rows,
full+h9 on gated rows; same weights. 074 = 073 + gated DTA logRidge alpha 0.3. Nothing fitted on test.
usage (from experiments/push-1300): python make_076.py -> ../../artifacts/submission-candidate-076.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-076.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p, c74 = rd('021'), rd('074'); G = np.load(_repo_io('../push-1200/fin-d1234n12.npz')); assert [str(i) for i in G['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); S = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
failed = R['failed'] | S['failed']; nzt = np.nan_to_num(G['noise'], nan=0); gate = (nzt >= 12) & ~failed; ng = (nzt < 12) & ~failed
T9 = np.load(_repo_io('fin-h9-tab-cpu.npz')); Tn = np.load(_repo_io('fin-nb-tab-cpu.npz')); assert [str(i) for i in T9['ids']] == order
full = {c: (rd(c) - 0.25 * p) / 0.75 for c in ('038', '039', '044')}
b0 = np.where(ng, np.mean([Tn[f'nb_{c}'] for c in full], 0), np.mean([full[c] for c in full], 0)); b9 = np.where(ng, np.mean([T9[f'nb_{c}'] for c in full], 0), np.mean([T9[f'full_{c}'] for c in full], 0))
cand = np.where(failed, c74, c74 + 0.384 * (b9 - b0)); assert np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c74; c61 = rd('061')
info = dict(candidate='076', formula='074 + 0.384*(H9 block - block) [H9: +cumulant p34, contrast (sigma 1) in all 12 TabPFN members]', change_rms_vs_074=float(np.sqrt((d ** 2).mean())),
            change_rms_nongated=float(np.sqrt((d[ng] ** 2).mean())), change_rms_gated=float(np.sqrt((d[gate] ** 2).mean())), change_rms_vs_061=float(np.sqrt(((cand - c61) ** 2).mean())), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-076.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
