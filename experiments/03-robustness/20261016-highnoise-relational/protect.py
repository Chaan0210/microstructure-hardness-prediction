"""082/076 conservation record; every experiment output stays beside this file."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,subprocess,sys,time,datetime,os
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
PRIOR=_repo_path(ROOT, 'experiments/plateau-break-20261016-2035')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main(action):
    start=time.monotonic()
    if action=='freeze':
        old=json.loads((_repo_path(PRIOR, 'protected.json')).read_text());old.update(json.loads((_repo_path(PRIOR, 'protected-upstream-extension.json')).read_text()))
        records={k:v['sha256'] for k,v in old.items()}
        # Latest measured result, retained prior research, and new inner076
        # anchors are read-only inputs not all present in the earlier inventory.
        extra=[_repo_path(ROOT, 'reports/candidate082-public-submission.json'),_repo_path(ROOT, 'artifacts/submission-candidate-082.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-076.csv')]
        extra+=list(PRIOR.rglob('*'))
        comp=_repo_path(ROOT, 'experiments/076-complement-20261016')
        for k in range(5):extra+=list((comp/f'teachers076/fold{k}-outer0').rglob('*'))
        extra += [comp/'retarget_teachers.py',comp/'probe.py']
        for p in extra:
            if p.is_file() and not any(x in p.parts for x in ['__pycache__','tmp','mpl-cache']):records[str(p.relative_to(ROOT))]=sha(p)
    else:records=json.loads((_repo_path(HERE, 'protected.json')).read_text())
    changed=[]
    for j,(p,h) in enumerate(records.items()):
        f=_repo_path(ROOT, p)
        if not f.is_file() or sha(f)!=h:changed.append(p)
        if j%30000==0:print(action,j,flush=True)
    assert not changed,changed
    env=dict(os.environ);env['GIT_OPTIONAL_LOCKS']='0'
    git=lambda *a:subprocess.check_output(['git',*a],cwd=_repo_io(ROOT),text=True,env=env)
    if action=='freeze':
        with (_repo_path(HERE, 'protected.json')).open('x') as f:json.dump(records,f,indent=2)
        code='import importlib.metadata as m,json,sys;print(json.dumps(dict(python=sys.version,executable=sys.executable,packages={d.metadata["Name"]:d.version for d in m.distributions()})))'
        record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=git('rev-parse','HEAD'),status=git('status','--porcelain=v1'),diff=git('diff','--binary'),staged=git('diff','--cached','--binary'),
            environment=json.loads(subprocess.check_output([str(_repo_path(ROOT, '.venv/bin/python')),'-B','-c',code],text=True)),
            incumbents={k:dict(csv_sha256=sha(_repo_path(ROOT, f'artifacts/submission-candidate-{k}.csv')),public_record=json.loads((_repo_path(ROOT, f'reports/candidate{k}-public-submission.json')).read_text())) for k in ['076','082']},
            seeds_and_full_reproduction_provenance='See immutable prior plateau-break-20261016-2035/{incumbent.json,environments.json,RUN.md,model-replay.json} plus round4-full073/final500/recipe.json.',
            files=len(records),seconds=time.monotonic()-start)
        with (_repo_path(HERE, 'starting-state.json')).open('x') as f:json.dump(record,f,indent=2)
    else:
        before=json.loads((_repo_path(HERE, 'starting-state.json')).read_text())
        assert before['head']==git('rev-parse','HEAD') and before['diff']==git('diff','--binary') and before['staged']==git('diff','--cached','--binary')
        with (_repo_path(HERE, 'preservation.json')).open('x') as f:json.dump(dict(all_unchanged=True,files=len(records),changed=changed,head_unchanged=True,tracked_and_staged_diffs_unchanged=True,seconds=time.monotonic()-start),f,indent=2)
    print(action,'COMPLETE',len(records),round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':main(sys.argv[1])
