"""Extra seeds of Study900's grain-set regressor (zero-shot object bank, identical config apart from the seed) for dev: outputs seed<k>/oof.npz."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import copy, importlib.util, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; S9 = _repo_path(HERE.parents[1], 'probabilistic-grains-900')
spec = importlib.util.spec_from_file_location('grain_set_run', _repo_path(HERE, 'run.py')); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
ORIG = copy.deepcopy(run.CONFIG); run.torch.set_num_threads(2)
bank, base, matched = run.load_inputs(_repo_path(S9, 'object-bank-train.npz'), _repo_path(S9, 'base.npz'))
for seed in [int(s) for s in sys.argv[1].split(',')]:
    run.CONFIG = {**copy.deepcopy(ORIG), 'seed': seed}; out = _repo_path(HERE, f'seed{seed}'); out.mkdir(parents=True, exist_ok=True)
    run.run_train(bank, base, matched, _repo_path(S9, 'object-bank-train.npz'), _repo_path(S9, 'base.npz'), out, 'cpu'); print('done', seed, flush=True)
