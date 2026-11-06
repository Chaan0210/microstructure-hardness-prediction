"""Sequential validation; no competing fit jobs and no silent retries."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import subprocess,json,time,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];PY=_repo_path(ROOT, '.venv/bin/python')
stages=[('evaluate',['evaluate.py','dense']),('dense-decision',['dense_decision.py']),('legacy',['legacy_replay.py']),('calibration',['calibration_diagnostic.py']),('repeat',['verify.py','repeat']),('log-models',['verify.py','models']),('assembly',['verify.py','assembly']),('dense-repeat',['dense_verify.py']),('trace',['trace_selection.py']),('reproduce',['reproduce.py'])]
assert json.loads((_repo_path(HERE, 'dense-complete.json')).read_text())['complete']
history=_repo_path(HERE, 'verification-stages.jsonl');past=[json.loads(x) for x in history.read_text().splitlines()] if history.exists() else []
for tag,args in stages:
 if any(r['stage']==tag and r['returncode']==0 for r in past):continue
 attempt=1+sum(r['stage']==tag for r in past);start=time.monotonic()
 with (_repo_path(HERE, f'final-{tag}-attempt{attempt}.log')).open('x') as f:result=subprocess.run([str(PY),'-B','-u',str(_repo_path(HERE, args[0])),*args[1:]],cwd=_repo_io(ROOT),stdout=f,stderr=subprocess.STDOUT)
 row=dict(stage=tag,attempt=attempt,seconds=time.monotonic()-start,returncode=result.returncode)
 with history.open('a') as f:f.write(json.dumps(row)+'\n')
 past.append(row);print(json.dumps(row),flush=True)
 if result.returncode:raise SystemExit('Stage failed; preserve the log and repair explicitly before resuming: '+tag)
with (_repo_path(HERE, 'verification-stage-times.json')).open('x') as f:json.dump(past,f,indent=2)
