from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-054.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p = rd('021'); c51 = rd('051')
T3 = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True); T6 = np.load(_repo_io('setZ/full500/predictions.npz'), allow_pickle=True)
assert [str(i) for i in T3['ids']] == order and [str(i) for i in T6['ids']] == order
S9 = (3 * T3['set_mean_raw'] + 6 * T6['set_mean_raw']) / 9
Gm = np.load(_repo_io('gmm-partner-test.npz'), allow_pickle=True); assert [str(i) for i in Gm['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); failed = R['failed'] | T3['failed'] | T6['failed']
cand = np.where(failed, p, 0.77 * c51 + 0.08 * S9 + 0.15 * Gm['pred'])
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate='054', formula='where(failed, p021, 0.77*cand051 + 0.08*grainSet9seed + 0.15*gmmRidge500)', dev_051=9.8190, dev_054=9.7637,
            change_rms_vs_053=float(np.sqrt(((cand - rd('053')) ** 2).mean())), n_failed=int(failed.sum()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-054.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
