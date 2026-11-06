"""Candidate 066 = 065 + 1[noise >= 12 & ~failed] * 0.16 * (mean_s dta_s - mean_s ctrl_s): the CNN term (021, weight 0.16) moved on noisy rows by the
paired effect of degradation-transfer augmentation on a TinyCNN128 ensemble (final_cnn_dta.py: both arms trained on the 500 train images, 'dta' adds
32 label-preserving degraded copies of clean train images per update; same seeds/streams otherwise). Same weight, nothing fitted on test data.
Dev (065 blend, 8 seeds x 3 partitions): see NOTES §10. usage: make_066.py DEV"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, json, glob, hashlib, numpy as np
from pathlib import Path
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-066.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
c65 = rd('065'); G = np.load(_repo_io('fin-d1to7n12-wal.npz'))
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); S = np.load(_repo_io('../probabilistic-grains-900/set_model/full500/predictions.npz'), allow_pickle=True)
gate = (np.nan_to_num(G['noise'], nan=0) >= 12) & ~(R['failed'] | S['failed'])
seeds = sorted(int(f.split('-s')[1][:-4]) for f in glob.glob('fincnn-dta-s*.npz') if Path(f.replace('-dta-', '-ctrl-')).exists())
L = lambda a: [np.load(_repo_io(f'fincnn-{a}-s{s}.npz')) for s in seeds]
for d in L('ctrl') + L('dta'): assert [str(i) for i in d['ids']] == order
delta = np.mean([d['pred'] for d in L('dta')], 0) - np.mean([d['pred'] for d in L('ctrl')], 0)
cand = c65 + 0.16 * gate * delta; assert np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate='066', formula='065 + 1[noise>=12 & ~failed]*0.16*(mean dta - mean ctrl TinyCNN128)', seeds=seeds, dev=float(sys.argv[1]), n_gated=int(gate.sum()),
            delta_gated_mean=float(delta[gate].mean()), delta_gated_sd=float(delta[gate].std()),
            change_rms_vs_065=float(np.sqrt(((cand - c65) ** 2).mean())), change_rms_vs_061=float(np.sqrt(((cand - rd('061')) ** 2).mean())), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-066.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
