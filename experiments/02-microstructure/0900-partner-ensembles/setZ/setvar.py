"""Structural variants of Study900's grain-set regressor (zero-shot object bank, dev protocol), fixed a priori:
  log   - fitted on log(hardness) (skip Ridge, normalisation, epoch selection all on the log scale); predictions exponentiated
  wide  - encoder 64, embedding 32, head 64 (else identical)
outputs var-<NAME>/seed<k>/oof.npz with 'prediction' on the hardness scale.
usage: setvar.py NAME SEED[,SEED...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import copy, importlib.util, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; S9 = _repo_path(HERE.parents[1], 'probabilistic-grains-900')
spec = importlib.util.spec_from_file_location('grain_set_run', _repo_path(HERE, 'run.py')); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
ORIG = copy.deepcopy(run.CONFIG); run.torch.set_num_threads(2)
name = sys.argv[1]; bank, base, matched = run.load_inputs(_repo_path(S9, 'object-bank-train.npz'), _repo_path(S9, 'base.npz'))
EXTRA = {'log': {}, 'wide': {'encoder_width': 64, 'embedding_dim': 32, 'head_width': 64}}[name]
if name == 'log':
    bank['y'] = np.log(bank['y']); base = {**base, 'y': np.log(base['y']), 'baseline': np.log(base['baseline']), 'p021': np.log(base['p021']), 'grain': np.log(base['grain'])}
    run.metrics = lambda b, p, r: ({'metrics': {}}, {})   # hardness-scale metrics are computed by blend_dev
for seed in [int(s) for s in sys.argv[2].split(',')]:
    run.CONFIG = {**copy.deepcopy(ORIG), **EXTRA, 'seed': seed}; out = _repo_path(HERE, f'var-{name}', f'seed{seed}'); out.mkdir(parents=True, exist_ok=True)
    run.run_train(bank, base, matched, _repo_path(S9, 'object-bank-train.npz'), _repo_path(S9, 'base.npz'), out, 'cpu')
    if name == 'log':
        o = dict(np.load(_repo_io(out / 'oof.npz'))); o['prediction'] = np.exp(o['prediction']); o['ridge'] = np.exp(o['ridge']); o['y'] = np.exp(o['y']); np.savez_compressed(_repo_io(out / 'oof.npz'), **o)
    print('done', seed, flush=True)
