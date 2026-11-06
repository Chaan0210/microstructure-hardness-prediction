"""Candidate 055 = 054 structure with every 98-column model on the noise-band-aligned features (qm_feats.py; label-free, fitted on the 500 train
rows only): 0.77*(0.2*best021 + 0.48*grainQM + 0.16*ridgeQM + 0.16*logRidgeQM) + 0.08*grainSet9 + 0.15*gmmQM.
grainQM = mean of the 038/039/044 TabPFN slots refitted on aligned features (final_qm.py tab 38/39/44), ridge/logRidge/gmm from final_qm.py.
The grain-set model (zero-shot object bank) is unchanged (aligning its inputs hurt on dev). Dev (eval_qm.py): 054 9.7636 -> 9.6685,
partitions +0.145/+0.088/+0.053.
usage (from experiments/push-900): python make_055.py   -> ../../artifacts/submission-candidate-055.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-055.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p = rd('021')
T = {s: np.load(_repo_io(f'qm-final-tab{s}.npz'), allow_pickle=True) for s in ('38', '39', '44')}
for t in T.values(): assert [str(i) for i in t['ids']] == order
gq = np.mean([T[s]['g'] for s in ('38', '39', '44')], 0)
Rq = np.load(_repo_io('qm-final-ridge.npz'), allow_pickle=True); Gq = np.load(_repo_io('qm-final-gmm.npz'), allow_pickle=True)
assert [str(i) for i in Rq['ids']] == order and [str(i) for i in Gq['ids']] == order
T3 = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True); T6 = np.load(_repo_io('setZ/full500/predictions.npz'), allow_pickle=True)
assert [str(i) for i in T3['ids']] == order and [str(i) for i in T6['ids']] == order
S9 = (3 * T3['set_mean_raw'] + 6 * T6['set_mean_raw']) / 9
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); failed = R['failed'] | T3['failed'] | T6['failed']
c51q = 0.2 * p + 0.48 * gq + 0.16 * Rq['ridge'] + 0.16 * Rq['logridge']
cand = np.where(failed, p, 0.77 * c51q + 0.08 * S9 + 0.15 * Gq['pred'])
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate='055', formula='where(failed, p021, 0.77*(0.2*p021 + 0.48*grainQM + 0.16*ridgeQM + 0.16*logRidgeQM) + 0.08*grainSet9seed + 0.15*gmmQM)',
            dev_054=9.7636, dev_055=9.6685, dev_partition_gain=[0.145, 0.088, 0.053],
            change_rms_vs_054=float(np.sqrt(((cand - rd('054')) ** 2).mean())), change_rms_vs_053=float(np.sqrt(((cand - rd('053')) ** 2).mean())),
            n_failed=int(failed.sum()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-055.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
