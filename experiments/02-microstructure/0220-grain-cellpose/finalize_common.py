"""Shared CSV writer/validator for candidates 024/025 (same checks as finalize.py)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, hashlib
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_train_test():
    Ftr = np.load(_repo_path(R, 'features-v3-train.npz')); Fte = np.load(_repo_path(R, 'features-v3-test.npz')); keys = list(Ftr['keys'])
    assert keys == list(Fte['keys'])
    lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
    return Ftr['X'], np.array([lab[i] for i in Ftr['ids']]), Fte['X'], list(Fte['ids']), keys

def write_candidate(cid, grain_test, tids, keys, Xte, W, MIN_OBJ, extra):
    out = _repo_path(P, f'artifacts/submission-candidate-{cid}.csv'); assert not out.exists(), 'refuse overwrite'
    p021 = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'artifacts/submission-candidate-021.csv')))}
    order = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/sample_submission.csv')))]
    pos = {i: k for k, i in enumerate(tids)}
    failed = np.nan_to_num(Xte[:, keys.index('n_obj')], nan=0) < MIN_OBJ
    base = np.array([p021[i] for i in order]); gg = np.array([grain_test[pos[i]] for i in order]); ff = np.array([failed[pos[i]] for i in order])
    gg = np.where(ff, base, gg); cand = (1 - W) * base + W * gg
    assert len(cand) == 1000 and np.isfinite(cand).all()
    with out.open('x', newline='') as f:
        w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
    back = list(csv.DictReader(open(_repo_io(out)))); assert [r['ID'] for r in back] == order and np.allclose([float(r['hardness']) for r in back], cand)
    c023 = np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'artifacts/submission-candidate-023.csv')))])
    s = dict(candidate=cid, weight=W, n_train=500, test_failed_segmentations=int(ff.sum()),
             pred_stats=dict(mean=float(cand.mean()), std=float(cand.std()), min=float(cand.min()), max=float(cand.max())),
             corr_best021_vs_grain_test=float(np.corrcoef(base, gg)[0, 1]), rms_change_vs_021=float(np.sqrt(np.mean((cand - base) ** 2))),
             rms_change_vs_023=float(np.sqrt(np.mean((cand - c023) ** 2))), csv=str(out.relative_to(P)), csv_sha256=sha(out),
             best021_csv_sha256=sha(_repo_path(P, 'artifacts/submission-candidate-021.csv')), uploaded=False, **extra)
    json.dump(s, open(_repo_path(R, f'candidate{cid}-summary.json'), 'x'), indent=1)
    m = {'candidate': cid, 'CSV_absolute_path': str(out.resolve()), 'CSV_sha256': s['csv_sha256'], 'summary': f'experiments/grain-cellpose-220/candidate{cid}-summary.json',
         'summary_sha256': sha(_repo_path(R, f'candidate{cid}-summary.json')), 'best021_CSV_sha256': s['best021_csv_sha256'], 'uploaded': False, 'cost': 0}
    (_repo_path(P, f'artifacts/submission-candidate-{cid}.manifest.json')).write_text(json.dumps(m, indent=2))
    print(json.dumps(s, indent=1))
