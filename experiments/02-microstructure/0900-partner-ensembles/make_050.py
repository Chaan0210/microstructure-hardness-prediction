"""Candidate 050 = 0.2*best021 + 0.8*(0.6*grain045 + 0.4*ridgeLog), ridgeLog = geometric mean over the 4 variant feature sets (push-800
feat-<v>-anghcor, 98 columns) of exp(Ridge(alpha 2) fitted on log(hardness)) with median imputation + standardisation, all fitted on the 500 train
rows only. Dev (diverse_dev.py ridge_a2_logy_var4, Study900 protocol): 045 9.9402 -> 9.7928, 048 9.8774. Failed images (n_obj < 20) keep best021.
usage (from experiments/push-900): python make_050.py   -> ../../artifacts/submission-candidate-050.csv (never overwrites)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, 'submission-candidate-050.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
p = rd('021'); g = np.mean([p + (rd(c) - p) / 0.75 for c in ('038', '039', '044')], 0)
R = np.load(_repo_io('../probabilistic-grains-900/ridge/final500/predictions.npz')); assert [str(i) for i in R['ids']] == order; failed = R['failed']
logs = []
for v in ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z'):
    Tr = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); Te = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-test.npz'), allow_pickle=True)
    assert list(Tr['keys']) == list(Te['keys']) and [str(i) for i in Te['ids']] == order
    y = np.log([lab[str(i)] for i in Tr['ids']]); m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(Tr['X'].astype(float), y)
    logs.append(m.predict(Te['X'].astype(float)))
ridge = np.exp(np.mean(logs, 0))
cand = np.where(failed, p, 0.2 * p + 0.8 * (0.6 * g + 0.4 * ridge))
assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); b45 = np.where(failed, p, p + 0.8 * (g - p))
info = dict(candidate='050', formula='where(n_obj<20, p021, 0.2*p021 + 0.8*(0.6*grain045 + 0.4*ridgeLog_a2_var4))', dev_045=9.940240, dev_048=9.877356, dev_050=9.7928,
            ridge_test_range=[float(ridge.min()), float(ridge.max())], ridge_vs_grain_rms=float(np.sqrt(np.mean(np.where(failed, 0, ridge - g) ** 2))),
            change_rms_vs_045=float(np.sqrt(((cand - b45) ** 2).mean())), failed=int(failed.sum()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, 'submission-candidate-050.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
