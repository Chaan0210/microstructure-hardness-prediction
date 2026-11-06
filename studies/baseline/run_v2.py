"""Run from the project root using .venv/bin/python."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
ROOT = _RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve() != (_repo_path(ROOT, ".venv")).resolve():
    raise SystemExit("Use this project's uv .venv: .venv/bin/python run.py ...")
sys.path.insert(0,str(_repo_path(ROOT, "src")))
from microstructure_v2.pipeline import main
if __name__ == "__main__":
    main()
