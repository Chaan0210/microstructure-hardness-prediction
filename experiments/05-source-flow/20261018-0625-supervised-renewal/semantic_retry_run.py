from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, subprocess, json, fcntl
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
tag = 'semantic-pfn-f0-c160-v2'
probe = json.loads((_repo_path(HERE, f'runs/{tag}/probe/probe.json')).read_text())
assert probe['optimizer_updates'] == 0 and probe['current_rss'] < 7*2**30 and probe['memory'] < 8*2**30
with (_repo_path(HERE, 'tmp/heavy-learning.lock')).open('a') as guard:
    fcntl.flock(guard, fcntl.LOCK_EX)
    with (_repo_path(HERE, 'semantic-pfn-c160-v2.log')).open('w') as log:
        subprocess.run([str(_repo_path(ROOT, 'experiments/tabpfn-v2/.venv/bin/python')), '-u', str(_repo_path(HERE, 'semantic_pfn_learning_v2.py')),
                        'train', '--tag', tag, '--context', '160'], cwd=_repo_io(ROOT),
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(_repo_path(HERE, 'tmp'))),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
result = json.loads((_repo_path(HERE, f'runs/{tag}/done.json')).read_text())
(_repo_path(HERE, 'semantic-stage-complete.json')).write_text(json.dumps(dict(completed_run=f'runs/{tag}',
                                                               original_failed_run='runs/semantic-pfn-f0-c160',
                                                               result=result, resource_probe=probe), indent=2))
print('SEMANTIC_STAGE_COMPLETE', json.dumps(result), flush=True)
