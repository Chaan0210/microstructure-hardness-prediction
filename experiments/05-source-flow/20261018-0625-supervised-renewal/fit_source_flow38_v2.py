"""Bounded local full-fit workflow for an already frozen source candidate."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, subprocess
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def run(out, python, script, args, label, lock_parent=False):
    command = [str(_repo_path(ROOT, python)), '-u', str(_repo_path(HERE, script)), *args]
    print('START', label, flush=True)
    with (out/f'{label}.log').open('w') as log:
        def dispatch():
            subprocess.run(command, cwd=_repo_io(ROOT), env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(_repo_path(HERE, 'tmp'))),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if lock_parent:
            with (_repo_path(HERE, 'tmp/heavy-learning.lock')).open('a') as guard:
                fcntl.flock(guard, fcntl.LOCK_EX); dispatch()
        else:
            dispatch()  # Inference/readout children acquire the same lock themselves.
    print('DONE', label, flush=True)


def main(a):
    out = Path(a.candidate).resolve(); assert out.is_relative_to(HERE)
    r = json.loads((out/'frozen-recipe.json').read_text()); assert r['status'].startswith('frozen_')
    repair = json.loads((out/'metadata-repair.json').read_text())
    assert repair['frozen_recipe_sha256'] == sha(out/'frozen-recipe.json')
    assert repair['new_inference_sha256'] == sha(_repo_path(HERE, 'source_flow38_infer_v2.py'))
    assert repair['new_workflow_sha256'] == sha(Path(__file__))
    for name, digest in r['source_hashes'].items():
        assert sha(_repo_path(HERE, name)) == digest
    for name, digest in r['training_input_hashes'].items():
        assert sha(_repo_path(HERE, name)) == digest
    assert sha(_repo_path(ROOT, r['initial_weights'])) == r['initial_weights_sha256']
    assert sha(_repo_path(ROOT, r['noise_training_file'])) == r['noise_training_sha256']
    if not (out/'full-fit-started.json').exists():
        assert datetime.datetime.now(datetime.timezone.utc) <= datetime.datetime.fromisoformat(r['final_fit_start_cutoff_utc']), 'Final-fit start cutoff reached; preserve recovery time.'
        (out/'full-fit-started.json').write_text(json.dumps(dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs_verified=True),indent=2))
    cellpose = 'experiments/grain-cellpose-220/.venv/bin/python'; tabpfn = 'experiments/tabpfn-v2/.venv/bin/python'
    for tag, seed in zip(r['encoder_tags'], r['encoder_training_seeds']):
        source = _repo_path(HERE, 'source-h300', tag)
        if not (source/'trained.json').exists():
            run(out,cellpose,'source_h300_v2.py',['train','--protocol','full500','--tag',tag,'--seed',str(seed)],tag+'-train',True)
        else:
            trained=json.loads((source/'trained.json').read_text());assert trained['complete'] and trained['model_sha256']==sha(source/'model.pt')
        run(out,cellpose,'source_flow38_infer_v2.py',['--candidate',str(out),'--encoder-tag',tag,'--split','train'],tag+'-train-features')
        if not (out/'models'/tag/'fit-complete.json').exists():
            run(out,tabpfn,'source_flow38_heads.py',['fit','--candidate',str(out),'--encoder-tag',tag],tag+'-heads')
        run(out,cellpose,'source_flow38_infer_v2.py',['--candidate',str(out),'--encoder-tag',tag,'--split','test'],tag+'-test-features')
        resume=source/'resume.pt'
        if resume.exists():
            size=resume.stat().st_size;resume.unlink()
            (source/'completed-optimizer-cleanup.json').write_text(json.dumps(dict(file='resume.pt',bytes=size,reason='Final model, complete train/test feature caches, fitted heads, recipe and curves retained; release redundant optimizer checkpoint for the next encoder.'),indent=2))
    if not (out/'csv-verification.json').exists():
        run(out,tabpfn,'source_flow38_heads.py',['predict','--candidate',str(out)],'predict')
    run(out,tabpfn,'source_flow38_heads.py',['replay','--candidate',str(out)],'replay')
    print('FROZEN_SOURCE_CANDIDATE_COMPLETE',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);main(p.parse_args())
