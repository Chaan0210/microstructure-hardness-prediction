"""Re-execute the recorded assembly scripts of the incumbent chain 061 -> 073 -> 074 -> 076 (push-1200/make_061.py, push-1300/make_07{3,4,6}.py)
unchanged except for the output location: every output goes to repro/chain/, and each step reads its predecessor from repro/chain/ (so the chain
is verified end to end), while older frozen inputs (053, 021, 038, 039, 044 CSVs and all component npz files) are read from their original
locations. Compares each rebuilt CSV byte-for-byte (SHA-256) with the stored artifact.
usage (from the project root): .venv/bin/python experiments/stepchange-20261016/repro_chain.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import hashlib, json, os, re
from pathlib import Path

ROOT = _RepoPath(__file__).resolve().parents[3]
import sys
OUT = _repo_path(Path(__file__).resolve().parent, 'repro', sys.argv[1] if len(sys.argv) > 1 else 'chain')   # argv[1]: output subfolder (post-experiment re-check: chain-post)
OUT.mkdir(parents=True, exist_ok=True)
steps = [('061', 'experiments/push-1200/make_061.py'), ('073', 'experiments/push-1300/make_073.py'),
         ('074', 'experiments/push-1300/make_074.py'), ('076', 'experiments/push-1300/make_076.py')]
res = {}
for cand, script in steps:
    src = (_repo_path(ROOT, script)).read_text()
    # outputs -> OUT (the scripts open them with mode 'x', so nothing can be overwritten)
    src = src.replace(f"out = A / 'submission-candidate-{cand}.csv'", f"out = Path(r'{OUT}') / 'submission-candidate-{cand}.csv'")
    src = src.replace(f"open(A / 'submission-candidate-{cand}.manifest.json', 'x')", f"open(Path(r'{OUT}') / 'submission-candidate-{cand}.manifest.json', 'x')")
    # reader: rebuilt predecessors from OUT, everything else from the original artifacts directory
    src = src.replace("rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(A / f'submission-candidate-{c}.csv'))])",
                      f"rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open((Path(r'{OUT}') if (Path(r'{OUT}') / f'submission-candidate-{{c}}.csv').exists() else A) / f'submission-candidate-{{c}}.csv'))])")
    assert src.count(str(OUT)) >= 4, f'patch failed for {script}'
    assert not re.search(r"open\(A / 'submission-candidate-\d+\.(csv|manifest\.json)', 'x'\)", src)
    cwd = os.getcwd(); os.chdir(_repo_path(ROOT, Path(script).parent))
    try:
        exec(compile(src, script, 'exec'), {'__name__': '__main__', '__file__': str(_repo_path(ROOT, script))})
    finally:
        os.chdir(_repo_io(cwd))
    new = OUT / f'submission-candidate-{cand}.csv'; old = _repo_path(ROOT, 'artifacts', f'submission-candidate-{cand}.csv')
    hn, ho = hashlib.sha256(new.read_bytes()).hexdigest(), hashlib.sha256(old.read_bytes()).hexdigest()
    res[cand] = dict(rebuilt_sha256=hn, stored_sha256=ho, identical=hn == ho)
    print(cand, 'IDENTICAL' if hn == ho else 'DIFFERENT', hn[:12], ho[:12], flush=True)
json.dump(res, open(_repo_io(OUT / 'chain-verification.json'), 'x'), indent=1)
