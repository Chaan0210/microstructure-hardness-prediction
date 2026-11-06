"""Generic 057/061-type candidate: 053 with the Ridge and log-Ridge components (each 0.128 of 053) replaced on test rows with pixel noise >= 12
by the same models fitted on the 500 train rows plus label-preserving degraded copies of train images (fin-<TAG>.npz from final_aug.py).
usage: make_aug.py CID FIN_TAG "description" DEV  -> ../../artifacts/submission-candidate-<CID>.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, json, hashlib, numpy as np
from pathlib import Path
cid, tag, desc, dev = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, f'submission-candidate-{cid}.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
c53 = rd('053'); N = np.load(_repo_io('fin-none.npz')); G = np.load(_repo_io(f'fin-{tag}.npz')); assert [str(i) for i in G['ids']] == order
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); T = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
failed = R['failed'] | T['failed']; gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~failed
cand = c53 + gate * (0.128 * (G['ridge'] - N['ridge']) + 0.128 * (G['logridge'] - N['logridge']))
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c53
info = dict(candidate=cid, formula='053 + 1[noise>=12 & ~failed]*0.128*((ridge_aug - ridge) + (logridge_aug - logridge))', augmentation=desc, fin=tag, dev=dev,
            n_fit_rows=int(G['n_fit']), ridge_alpha=float(G['alpha']), n_gated=int(gate.sum()), change_rms_all=float(np.sqrt((d ** 2).mean())),
            change_rms_gated=float(np.sqrt((d[gate] ** 2).mean())), change_mean_gated=float(d[gate].mean()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, f'submission-candidate-{cid}.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
