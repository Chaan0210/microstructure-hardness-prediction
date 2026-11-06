"""Candidate = BASE + A*(DIR - BASE) for two submitted candidate CSVs (exact linear mix). usage: make_mix.py CID BASE DIR A [note]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, json, hashlib, numpy as np
from pathlib import Path
cid, base, dirc, a = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); note = sys.argv[5] if len(sys.argv) > 5 else ''
A = Path(_repo_io('../../artifacts')); out = _repo_path(A, f'submission-candidate-{cid}.csv'); assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(A, f'submission-candidate-{c}.csv')))])
b, d = rd(base), rd(dirc); cand = b + a * (d - b); assert np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest()
info = dict(candidate=cid, formula=f'{base} + {a}*({dirc} - {base})', change_rms_vs_base=float(np.sqrt(((cand - b) ** 2).mean())), note=note, CSV_sha256=sha, uploaded=False)
json.dump(info, open(_repo_path(A, f'submission-candidate-{cid}.manifest.json'), 'x'), indent=1); print(json.dumps(info, indent=1))
