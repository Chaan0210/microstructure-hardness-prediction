"""Replay the exact frozen implementation in a separate local output directory."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import argparse, json, shutil, sys
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['fit', 'features', 'predict', 'replay'])
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    out = Path(a.output_dir).resolve()
    assert out.is_relative_to(HERE) and out != HERE and out != _repo_path(HERE, 'candidate-flow46-v1')
    if a.action == 'fit':
        out.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(_repo_path(HERE, 'candidate-flow46-v1/frozen-recipe.json'), out/'frozen-recipe.json')
    else:
        assert (out/'frozen-recipe.json').exists()
    if a.action == 'features':
        import flow_test_inference as module
        module.OUT = out
        module.main()
    else:
        import flow_candidate_finalize as module
        from threadpoolctl import threadpool_limits
        module.OUT = out
        module.torch.set_num_threads(2); module.torch.set_num_interop_threads(1)
        with threadpool_limits(2):
            if a.action == 'fit':
                module.fit()
            else:
                module.predict(a.action == 'replay')


if __name__ == '__main__':
    main()
