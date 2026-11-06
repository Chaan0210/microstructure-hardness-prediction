from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-049.csv'); assert not out.exists()
rd = lambda c: {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))}
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
p = np.array([rd('021')[i] for i in order]); g = np.mean([p + (np.array([rd(c)[i] for i in order]) - p) / 0.75 for c in ('038', '039', '044')], 0)
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); assert [str(i) for i in R['ids']] == order
assert np.allclose(R['baseline045'], p + 0.8 * (g - p)) and np.allclose(R['p021'], p)
failed = R['failed']; cand = np.where(failed, p, 0.2 * p + 0.8 * (0.6 * g + 0.4 * R['raw_ridge']))
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); b45 = R['baseline045']
info = dict(candidate='049', formula='where(n_obj<20, p021, 0.2*p021 + 0.8*(0.6*grain045 + 0.4*ridge500))', dev_045=9.940240, dev_049=9.860873,
            change_rms_vs_045=float(np.sqrt(((cand - b45) ** 2).mean())), failed=int(failed.sum()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-049.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
