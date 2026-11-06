"""Reproduce the frozen source candidate in a fresh, isolated local directory.

states: saved states and features; heads: refit heads from saved train features;
features: regenerate test-image features from saved encoders; full: retrain both
encoders and heads. This is an explicit reproduction command, not model search.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
ORIGINAL = _repo_path(HERE, 'candidate-source-flow38-v1')
CELLPOSE = _repo_path(ROOT, 'experiments/grain-cellpose-220/.venv/bin/python')
TABPFN = _repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def run(out, python, script, args, label, parent_lock=False):
    print('START',label,flush=True)
    def launch():
        with (out/f'{label}.log').open('w') as log:
            subprocess.run([str(python),'-u',str(_repo_path(HERE, script)),*args],cwd=_repo_io(ROOT),
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(_repo_path(HERE, 'tmp'))),
                stdout=log,stderr=subprocess.STDOUT,check=True)
    if parent_lock:
        with (_repo_path(HERE, 'tmp/heavy-learning.lock')).open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX);launch()
    else:
        launch()


def main(a):
    assert (_repo_path(ORIGINAL, 'replay-verification.json')).exists(), 'Wait for the original candidate to finish.'
    out=Path(a.output_dir).resolve()
    assert out.is_relative_to(HERE) and out.parent==HERE and not out.exists()
    recipe=json.loads((_repo_path(ORIGINAL, 'frozen-recipe.json')).read_text())
    repair=json.loads((_repo_path(ORIGINAL, 'metadata-repair.json')).read_text())
    assert sha(_repo_path(ORIGINAL, 'frozen-recipe.json'))==repair['frozen_recipe_sha256']
    for name,digest in recipe['source_hashes'].items():
        assert sha(_repo_path(HERE, name))==digest
    assert sha(_repo_path(HERE, 'source_flow38_infer_v2.py'))==repair['new_inference_sha256']
    if a.mode=='full':
        assert shutil.disk_usage(HERE).free >= 32*2**30, 'Full reproduction requires32GiB free; do not delete protected artifacts to make room.'
        for name,digest in recipe['training_input_hashes'].items():
            assert sha(_repo_path(HERE, name))==digest
        assert sha(_repo_path(ROOT, recipe['initial_weights']))==recipe['initial_weights_sha256']
        assert sha(_repo_path(ROOT, recipe['noise_training_file']))==recipe['noise_training_sha256']
        recipe['encoder_tags']=[f'reproduce-{out.name}-s{seed}' for seed in recipe['encoder_training_seeds']]
        assert all(not (_repo_path(HERE, 'source-h300', tag)).exists() for tag in recipe['encoder_tags'])
    out.mkdir()
    recipe_path=out/'frozen-recipe.json'
    if a.mode=='full':
        recipe_path.write_text(json.dumps(recipe,indent=2)+'\n')
    else:
        recipe_path.write_bytes((_repo_path(ORIGINAL, 'frozen-recipe.json')).read_bytes())
    clone_repair={k:repair[k] for k in ['original_inference_sha256','new_inference_sha256']}
    clone_repair.update(frozen_recipe_sha256=sha(recipe_path),reproduction_of=str(ORIGINAL),
                       change='Same metadata-only check repair; original learning/forward/descriptor recipe retained.')
    (out/'metadata-repair.json').write_text(json.dumps(clone_repair,indent=2)+'\n')
    (out/'reproduction-contract.json').write_text(json.dumps(dict(mode=a.mode,
        original_recipe_sha256=sha(_repo_path(ORIGINAL, 'frozen-recipe.json')),output_recipe_sha256=sha(recipe_path),
        original_directory=str(ORIGINAL),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        only_full_mode_recipe_change='Encoder output tags; all learning settings and seeds unchanged.',
        full_mode_uses_original_training_only=True,no_submission_or_selection_change=True),indent=2)+'\n')
    if a.mode in ['states','heads']:
        (out/'features').symlink_to(_repo_path(ORIGINAL, 'features'),target_is_directory=True)
    if a.mode in ['states','features']:
        (out/'models').symlink_to(_repo_path(ORIGINAL, 'models'),target_is_directory=True)
    for tag,seed in zip(recipe['encoder_tags'],recipe['encoder_training_seeds']):
        if a.mode=='full':
            run(out,CELLPOSE,'source_h300_v2.py',['train','--protocol','full500','--tag',tag,'--seed',str(seed)],tag+'-train',True)
            run(out,CELLPOSE,'source_flow38_infer_v2.py',['--candidate',str(out),'--encoder-tag',tag,'--split','train'],tag+'-train-features')
        if a.mode in ['full','heads']:
            run(out,TABPFN,'source_flow38_heads.py',['fit','--candidate',str(out),'--encoder-tag',tag],tag+'-heads')
        if a.mode in ['full','features']:
            run(out,CELLPOSE,'source_flow38_infer_v2.py',['--candidate',str(out),'--encoder-tag',tag,'--split','test'],tag+'-test-features')
    for action in ['predict','replay']:
        run(out,TABPFN,'source_flow38_heads.py',[action,'--candidate',str(out)],action)
    print('REPRODUCTION_COMPLETE',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['states','heads','features','full'],required=True)
    p.add_argument('--output-dir',required=True)
    main(p.parse_args())
