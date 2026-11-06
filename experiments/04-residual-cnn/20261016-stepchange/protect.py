"""Protected-artifact record and verification for the incumbent (candidate 076) and its upstream pipeline.
usage (from the project root):
  python experiments/stepchange-20261016/protect.py snapshot   -> record/protected-sha256.json, record/tree-stat.json
  python experiments/stepchange-20261016/protect.py verify     -> record/verify-<time>.json (exit 1 on any change)
Reads only; writes only inside experiments/stepchange-20261016/record."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import hashlib, json, os, sys, time
from pathlib import Path

ROOT = _RepoPath(__file__).resolve().parents[3]
ME = Path(__file__).resolve().parent
REC = _repo_path(ME, 'record')
# directories whose every file is SHA-256 hashed (the incumbent's upstream closure: features, U-Net weights/maps, TabPFN outputs, partners,
# degraded-copy rows, CNN021/round4 components, folds, data, all candidate CSVs/manifests)
HASH_DIRS = ['data', 'experiments/push-1300', 'experiments/push-1200', 'experiments/probabilistic-grains-900', 'experiments/push-800',
             'experiments/push-900', 'experiments/push-400', 'experiments/push-300', 'experiments/resolution-216', 'experiments/resolution-repartition-217',
             'experiments/rmse-research-20261015', 'experiments/078-080-update-20261016', 'experiments/grain-cellpose-220/features',
             'reports']
HASH_GLOBS = ['artifacts/submission-candidate-*', 'artifacts/test-prediction-features.npz']
# excluded from the tree snapshot: this experiment, git internals, and the directory another session is writing right now
SKIP = {'.git', 'experiments/stepchange-20261016', 'experiments/076-complement-20261016', 'experiments/push-1500', '__pycache__'}


def sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(_repo_io(p), 'rb') as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def hashed_files():
    files = []
    for d in HASH_DIRS:
        base = _repo_path(ROOT, d)
        if base.exists():
            files += [p for p in base.rglob('*') if p.is_file() and not p.is_symlink() and '.venv' not in p.parts and '__pycache__' not in p.parts]
    for g in HASH_GLOBS:
        files += [p for p in ROOT.glob(g) if p.is_file()]
    return sorted(set(files))


def tree_stat():
    out = {}
    for dp, dn, fn in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        dn[:] = [d for d in dn if (os.path.join(rel, d) if rel != '.' else d) not in SKIP and d not in ('.git', '__pycache__')]
        for f in fn:
            p = os.path.join(dp, f)
            try:
                st = os.lstat(p)
            except FileNotFoundError:
                continue
            out[os.path.relpath(p, ROOT)] = [st.st_size, st.st_mtime_ns]
    return out


if __name__ == '__main__':
    mode = sys.argv[1]
    t0 = time.time()
    if mode == 'snapshot':
        REC.mkdir(exist_ok=True)
        assert not (_repo_path(REC, 'protected-sha256.json')).exists(), 'snapshot already taken'
        H = {str(p.relative_to(ROOT)): sha(p) for p in hashed_files()}
        json.dump(H, open(_repo_path(REC, 'protected-sha256.json'), 'x'), indent=0)
        T = tree_stat()
        json.dump(T, open(_repo_path(REC, 'tree-stat.json'), 'x'))
        print(f'hashed {len(H)} files, tree {len(T)} files, {time.time() - t0:.0f}s')
    elif mode == 'snapshot2':   # 2026-10-16 21:45: after 082 became the submitted/auto-final candidate; same closure re-hashed (adds reports/candidate082-public-submission.json)
        assert not (_repo_path(REC, 'protected-sha256-v2.json')).exists()
        H = {str(p.relative_to(ROOT)): sha(p) for p in hashed_files()}; H0 = json.load(open(_repo_path(REC, 'protected-sha256.json')))
        assert all(H.get(k) == v for k, v in H0.items()), 'v1-protected file changed before v2 snapshot'
        json.dump(H, open(_repo_path(REC, 'protected-sha256-v2.json'), 'x'), indent=0); print(f'v2: hashed {len(H)} files ({len(H) - len(H0)} new since v1), {time.time() - t0:.0f}s')
    elif mode == 'verify':
        H0 = json.load(open(_repo_path(REC, 'protected-sha256-v2.json' if (REC / 'protected-sha256-v2.json').exists() else 'protected-sha256.json'))); T0 = json.load(open(_repo_path(REC, 'tree-stat.json')))
        changed = [k for k, v in H0.items() if not (_repo_path(ROOT, k)).exists() or sha(_repo_path(ROOT, k)) != v]
        T1 = tree_stat()
        modified = [k for k, v in T0.items() if k not in T1 or T1[k] != v]
        added = sorted(set(T1) - set(T0))
        res = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), n_hashed=len(H0), hash_changed=changed, n_tree=len(T0), tree_modified_or_deleted=modified,
                   tree_added=added, seconds=round(time.time() - t0, 1))
        json.dump(res, open(_repo_path(REC, f'verify-{time.strftime("%Y%m%d-%H%M%S")}.json'), 'x'), indent=1)
        print(json.dumps({k: (v if not isinstance(v, list) else (len(v), v[:20])) for k, v in res.items()}, indent=1))
        sys.exit(1 if changed or modified else 0)
