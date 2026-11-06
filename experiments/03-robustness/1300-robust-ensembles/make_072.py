"""Candidate 072 = 071 + half of the TA4 slot replacement of 069 on noisy rows (noise >= 12, not failed): 071 + 0.5 * 0.128 * sum over slots
(TA4 slot - original slot). Half weight because the TA4 component has strong train-only support (dev LM P 0.87-0.94, holdout -4.42 SE 4.63) but a
neutral Public reading (069: +0.019, SE 0.034): shrinking it halves the variance and the n-dependent test weight shift (K_eff 1.50 -> ~1.25).
Dev (vs 061): +0.047 (+0.083/+0.033/+0.027), non-stretch +0.039, LM noisy dMSE -1.60 (SE 0.95, P 0.95), 9/11 stripes. Nothing fitted on test."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-072.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
c61, c69, c71 = rd('061'), rd('069'), rd('071')
cand = c71 + 0.5 * (c69 - c61)   # 069 - 061 = gated TA4 slot change exactly (make_069.py)
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c61
info = dict(candidate='072', formula='071 + 0.5*(069 - 061)  [= 061 + WLS partners + DTA gate 11 + half TA4 slots on noise>=12]', dev_vs_061=0.0474,
            change_rms_vs_061=float(np.sqrt((d ** 2).mean())), change_rms_vs_071=float(np.sqrt(((cand - c71) ** 2).mean())), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-072.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
