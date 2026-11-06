"""Candidates 080 = 078 + residual-CNN correction, 081 = 079 + residual-CNN correction.
Correction = round4-full073 final500 'addition' candidate minus 073 (auxiliary residual CNN, 2 seeds, trained on 500 train images, production weight
0.5468 on noise >= 12 & ~failed rows only; weights = mean of five B-only inner calibrations; nothing fitted on test). The CNN correction is additive to the
H9 / 074 / sigma-2/4 changes on dev (original partition: +0.231 on 073, +0.240 on 078, +0.241 on 079; non-stretch +0.212, LM noisy dMSE -11.6).
usage (from experiments/push-1300): python make_080.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); R4 = Path(_repo_io('../rmse-research-20261015/round4-full073/final500'))
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda p: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_io(p)))])
corr = rd(_repo_path(R4, 'candidate-aux-noise-addition.csv')) - rd(_repo_path(A, 'submission-candidate-073.csv'))
for new, base in (('080', '078'), ('081', '079')):
    out = _repo_path(A, f'submission-candidate-{new}.csv'); assert not out.exists()
    c = rd(_repo_path(A, f'submission-candidate-{base}.csv')); cand = c + corr; assert np.isfinite(cand).all()
    with out.open('x', newline='') as f:
        w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    info = dict(candidate=new, formula=f'{base} + (round4 aux residual-CNN addition - 073) on noise>=12 rows', n_changed=int((np.abs(corr) > 1e-9).sum()), change_rms_vs_base=float(np.sqrt((corr ** 2).mean())),
                change_mean_changed=float(corr[np.abs(corr) > 1e-9].mean()), correction_source_sha256=hashlib.sha256((_repo_path(R4, 'candidate-aux-noise-addition.csv')).read_bytes()).hexdigest(), CSV_sha256=sha, uploaded=False)
    json.dump(info, open(_repo_path(A, f'submission-candidate-{new}.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
