"""Read-only completed204 verification; never train or infer."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import sys,json,hashlib
R=Path.cwd();O=Path(__file__).resolve().parent;assert R==_RepoPath(__file__).resolve().parents[3] and Path(sys.prefix)==_repo_path(R, '.venv')
def H(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def J(p):return json.loads(p.read_text())
m=J(_repo_path(O, 'artifact-manifest.json'));checks=0
for rel,h in m['files'].items():assert H(_repo_path(R, rel))==h,rel;checks+=1
p=J(_repo_path(O, 'plan.json'));pres=J(_repo_path(O, 'preservation.json'));changed={'README.md','reports/strategy-experiment-register.csv'}
for rel,h in {**p['inputs_sha256'],**p['protected_sha256']}.items():
 if rel not in changed:assert H(_repo_path(R, rel))==h,rel
for rel,h in pres['final_mutable_sha256'].items():assert H(_repo_path(R, rel))==h,rel
assert (_repo_path(R, 'README.md')).read_bytes().endswith((_repo_path(O, 'README.before.md')).read_bytes());assert (_repo_path(R, 'reports/strategy-experiment-register.csv')).read_bytes().startswith((_repo_path(O, 'strategy-experiment-register.before.csv')).read_bytes())
assert J(_repo_path(O, 'training-complete.json'))['main_fits']==540 and J(_repo_path(O, 'training-complete.json'))['validation_fits']==48
for name in ['preflight.json','model-validation.json','downstream-validation.json','score-validation.json']:assert J(_repo_path(O, name))['passed']
assert H(_repo_path(R, 'artifacts/submission-candidate-021.csv'))=='7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b'
print(json.dumps({'passed':True,'artifact_files_verified':checks,'new_training':0,'new_inference':0,'best021_sha256':H(_repo_path(R, 'artifacts/submission-candidate-021.csv'))}))
