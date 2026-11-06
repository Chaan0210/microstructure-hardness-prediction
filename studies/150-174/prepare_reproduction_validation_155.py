"""Create isolated package copy and contract checker; no experiments or fitting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import shutil,hashlib,json
from pathlib import Path
R=_RepoPath(__file__).resolve().parents[1];P=_repo_path(R, 'deliverables/reproduction-018-v155');I=_repo_path(R, 'deliverables/verification-155/package')
assert not I.exists();I.parent.mkdir(exist_ok=False);shutil.copytree(_repo_io(P),_repo_io(I))
(_repo_path(I, 'check_contract.py')).write_text('''import sys,ast,importlib.metadata,socket,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runtime as r
r.verify_assets();r.inference_fit_guard();env=r.read(r.ROOT/'environment.json');assert {k:importlib.metadata.version(k) for k in env['packages']}==env['packages']
for p in r.ROOT.rglob('*.py'):ast.parse(p.read_text())
blocked=0
for call in [lambda:socket.create_connection(('example.invalid',443)),lambda:r.residual_models.fit(None,None,None,None),lambda:r.torch.tensor(1.,requires_grad=True).backward()]:
 try:call()
 except RuntimeError:blocked+=1
assert blocked==3
project=r.ROOT.parents[2];bad=[]
for mod in list(sys.modules.values()):
 f=getattr(mod,'__file__',None)
 if f:
  p=Path(f).resolve()
  if p.is_relative_to(project) and not p.is_relative_to(r.ROOT) and not p.is_relative_to(project/'.venv'):bad.append(str(p.relative_to(project)))
assert not bad,bad
r.write(r.ROOT/'contract.json',{'passed':True,'source_imports_outside_package':bad,'network_fit_backward_guards':True,'environment_versions_exact':True,'training_driver_verified':False});print('isolated contract passed')
''')
print(str(I.relative_to(R)))
