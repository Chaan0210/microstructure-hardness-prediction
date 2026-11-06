"""Candidate 023 = 0.5 * best021 + 0.5 * grain model (fallback to best021 when segmentation failed).

Same procedure as validate_v2.py / plan-v2.json: fixed w=0.5, log1p(clip) on all columns, model family chosen by
5-fold CV on the training data only (here all 500 train images), then refit on all 500.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib, sys, warnings
import numpy as np
from pathlib import Path
from PIL import Image
warnings.filterwarnings('ignore')
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(R))
from features import feature_matrix
from models_v2 import W, MIN_OBJ, select_and_fit
from sklearn.model_selection import KFold

OUT = _repo_path(P, 'artifacts/submission-candidate-023.csv')
assert json.load(open(_repo_path(R, 'validation-v2.json')))['passed'], 'v2 validation gates not passed'
assert not OUT.exists(), 'refuse overwrite'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

# train
F = np.load(_repo_path(R, 'features-train.npz')); keys = list(F['keys'])
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
Xtr = F['X']; ytr = np.array([lab[i] for i in F['ids']])
folds500 = np.zeros(len(ytr), int)
for k, (_, v) in enumerate(KFold(5, shuffle=True, random_state=20261007).split(Xtr)): folds500[v] = k
family, family_scores, predict, model = select_and_fit(Xtr, ytr, folds500)

# test features
tf = _repo_path(R, 'features-test.npz')
if not tf.exists():
    z = np.load(_repo_path(R, 'masks-test.npz'))
    imgs = [np.array(Image.open(_repo_path(P, f'data/test/{i}.png'))) for i in z['ids']]
    Xte, kt = feature_matrix(imgs, z['masks'])
    assert kt == keys, 'feature keys differ'
    np.savez(_repo_io(tf), ids=z['ids'], X=Xte, keys=np.array(kt))
T = np.load(_repo_io(tf)); Xte, tids = T['X'], list(T['ids'])
g = predict(Xte)
nobj = np.nan_to_num(Xte[:, keys.index('n_obj')], nan=0); failed = nobj < MIN_OBJ

# best021 test predictions (protected file, read only)
p021 = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'artifacts/submission-candidate-021.csv')))}
order = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/sample_submission.csv')))]
pos = {i: k for k, i in enumerate(tids)}
base = np.array([p021[i] for i in order]); gg = np.array([g[pos[i]] for i in order]); ff = np.array([failed[pos[i]] for i in order])
gg = np.where(ff, base, gg)
cand = (1 - W) * base + W * gg
assert len(cand) == 1000 and np.isfinite(cand).all()
with OUT.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
# reload validation
back = list(csv.DictReader(open(_repo_io(OUT))))
assert [r['ID'] for r in back] == order and np.allclose([float(r['hardness']) for r in back], cand)
summary = dict(candidate='023', formula=f'{1-W:.2f}*best021 + {W:.2f}*grain (fallback best021 if n_obj<{MIN_OBJ})', weight=W, family=family, family_cv_rmse=family_scores,
               model_params=(model.best_params_ if hasattr(model, 'best_params_') else None), n_train=int(len(ytr)), test_failed_segmentations=int(ff.sum()),
               pred_stats=dict(mean=float(cand.mean()), std=float(cand.std()), min=float(cand.min()), max=float(cand.max())),
               best021_stats=dict(mean=float(base.mean()), std=float(base.std())), grain_test_stats=dict(mean=float(gg.mean()), std=float(gg.std())),
               corr_best021_vs_grain_test=float(np.corrcoef(base, gg)[0, 1]),
               rms_change_vs_best021=float(np.sqrt(np.mean((cand - base) ** 2))),
               dev_evidence='validation-v2.json: pooled 3-partition OOF 12.815->12.240 (gain 0.575, conditional CI [0.156,0.990]), all gates passed; same 400 images, not independent.',
               csv=str(OUT.relative_to(P)), csv_sha256=sha(OUT), best021_csv_sha256=sha(_repo_path(P, 'artifacts/submission-candidate-021.csv')),
               weights_sha256='e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2', uploaded=False)
json.dump(summary, open(_repo_path(R, 'candidate023-summary.json'), 'x'), indent=1)
print(json.dumps(summary, indent=1))
