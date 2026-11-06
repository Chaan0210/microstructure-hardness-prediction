"""Run last, after all writers exit; hash the completed experiment record."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
if __name__=='__main__':
    files={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','tmp','mpl-cache']) and p.name!='delivery-hashes.json'}
    (_repo_path(HERE, 'delivery-hashes.json')).write_text(json.dumps(files,indent=2)+'\n')
    assert all(sha(_repo_path(HERE, p))==h for p,h in files.items())
    print('FINAL DELIVERABLE MANIFEST VERIFIED',len(files),'files')
