"""Verify the 082/076 dependency set against the existing stepchange SHA-256 record (no re-hash of the whole project).
usage: protect.py [start|end] -> writes protection-{start,end}.json in this directory."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import hashlib, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = _RepoPath(__file__).resolve().parents[3]
REC = _repo_path(ROOT, 'experiments/stepchange-20261016/record/protected-sha256-v2.json')
PATS = [r'^artifacts/submission-candidate-0(21|38|39|44|53|61|73|74|76|78|80|82)\.', r'^experiments/push-1300/(fin-|make_0|final_h9)',
        r'^experiments/push-1200/(fin-|make_061)', r'^experiments/probabilistic-grains-900/(ridge/final500|set_model/full500)/predictions',
        r'^experiments/078-080-update-20261016/', r'^experiments/rmse-research-20261015/round4-full073/final500/',
        r'^data/train\.csv', r'^experiments/rmse-research-20261015/(inputs|scope|consistency-strict)\.npz',
        r'^experiments/push-1300/abl/(dev|feat)-h9', r'^experiments/076-complement-20261016/baseline076', r'^experiments/highnoise-ssl-20261016-2312/oof\.npz']
rec = json.loads(REC.read_text())
sel = {k: v for k, v in rec.items() if any(re.search(p, k) for p in PATS)}
def sha(p):
    h = hashlib.sha256()
    with open(_repo_io(p), 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
bad = {k: 'missing' if not (_repo_path(ROOT, k)).exists() else 'changed' for k, v in sel.items() if not (_repo_path(ROOT, k)).exists() or sha(_repo_path(ROOT, k)) != v}
extra = {}
for k in ['experiments/076-complement-20261016/baseline076.npz', 'experiments/highnoise-ssl-20261016-2312/oof.npz']:
    if k not in sel: extra[k] = sha(_repo_path(ROOT, k))
out = dict(record=str(REC.relative_to(ROOT)), n_checked=len(sel), n_bad=len(bad), bad=bad,
           sha_076=sha(_repo_path(ROOT, 'artifacts/submission-candidate-076.csv')), sha_082=sha(_repo_path(ROOT, 'artifacts/submission-candidate-082.csv')), unrecorded_inputs=extra)
tag = sys.argv[1] if len(sys.argv) > 1 else 'check'
(_repo_path(HERE, f'protection-{tag}.json')).write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('n_checked', 'n_bad', 'sha_076', 'sha_082')}, indent=1)); assert not bad
