"""Rehash every protected artifact; record final provenance without Git writes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import hashlib,json,time,subprocess,os
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    start=time.monotonic();protected=json.loads((_repo_path(HERE, 'protected.json')).read_text())
    extension=_repo_path(HERE, 'protected-upstream-extension.json')
    if extension.exists():protected.update(json.loads(extension.read_text()))
    missing=[];changed=[];size=0
    for j,(name,entry) in enumerate(protected.items()):
        p=_repo_path(ROOT, name)
        if not p.is_file():missing.append(name);continue
        size+=entry['size']
        if sha(p)!=entry['sha256']:changed.append(name)
        if j%30000==0:print('verified',j,flush=True)
    assert not missing and not changed,(missing,changed)
    env=dict(os.environ);env['GIT_OPTIONAL_LOCKS']='0'
    git=lambda *a:subprocess.check_output(['git',*a],cwd=_repo_io(ROOT),text=True,env=env)
    before=json.loads((_repo_path(HERE, 'repository.json')).read_text())
    record=dict(all_unchanged=True,files_verified=len(protected),bytes_verified=size,changed=changed,missing=missing,
        git_head_unchanged=before['head']==git('rev-parse','HEAD'),tracked_diff_unchanged=before['diff']==git('diff','--binary'),
        staged_diff_unchanged=before['staged_diff']==git('diff','--cached','--binary'),final_status=git('status','--porcelain=v1'),
        seconds=time.monotonic()-start)
    assert record['git_head_unchanged'] and record['tracked_diff_unchanged'] and record['staged_diff_unchanged']
    with (_repo_path(HERE, 'preservation.json')).open('x') as f:json.dump(record,f,indent=2)
    print('PRESERVED',len(protected),'files',size,'bytes',round(time.monotonic()-start,1),'seconds',flush=True)
if __name__=='__main__':main()
