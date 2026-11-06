from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib,subprocess,urllib.request,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
 with (_repo_path(R, n)).open('x') as f:json.dump(v,f,indent=2)
old=json.loads((_repo_path(P, 'experiments/tabpfn-v2/authorization-and-preservation.json')).read_text())
protected={**old['protected'],**old['original_environment_metadata_sha256']}
for n,h in protected.items():assert sha(_repo_path(P, n))==h,n
write('preservation.json',protected)
for n in ['cache','config','tmp','wheels','weights','licenses','uv-cache']:(_repo_path(R, n)).mkdir()
env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','UV_CACHE_DIR':str(_repo_path(R, 'uv-cache')),'UV_NO_CONFIG':'1','UV_PYTHON_DOWNLOADS':'never','PYTHONDONTWRITEBYTECODE':'1','XDG_CACHE_HOME':str(_repo_path(R, 'cache')),'XDG_CONFIG_HOME':str(_repo_path(R, 'config')),'TMPDIR':str(_repo_path(R, 'tmp'))}
subprocess.run(['/opt/homebrew/bin/uv','venv','--no-project','--python',str(_repo_path(P, '.venv/bin/python')),str(_repo_path(R, '.venv'))],env=env,check=True)
# APFS copy-on-write clone of approved existing dependencies, no shared writable paths.
src=_repo_path(P, '.venv/lib/python3.13/site-packages');dst=_repo_path(R, '.venv/lib/python3.13/site-packages')
for p in src.iterdir():
 if p.name.startswith('_virtualenv') or p.name=='__pycache__':continue
 subprocess.run(['/bin/cp','-cR',str(p),str(_repo_path(dst, p.name))],check=True,env=env)
url='https://pypi.org/pypi/torchvision/0.29.0/json'
with urllib.request.urlopen(url,timeout=30) as f:meta=json.load(f)
w=next(x for x in meta['urls'] if x['filename']=='torchvision-0.29.0-cp313-cp313-macosx_14_0_arm64.whl')
def fetch(url,p,maxbytes,expected=None):
 req=urllib.request.Request(url,headers={'User-Agent':'local-approved-research/185'})
 with urllib.request.urlopen(req,timeout=60) as response,p.open('xb') as out:
  assert response.status==200;count=0
  while b:=response.read(1048576):
   count+=len(b);assert count<=maxbytes;out.write(b)
 h=sha(p)
 if expected:assert h.startswith(expected),(p,h)
 return {'url':url,'bytes':count,'sha256':h,'path':str(p.relative_to(P))}
wheel=fetch(w['url'],_repo_path(R, 'wheels', w['filename']),2000000,w['digests']['sha256'])
subprocess.run(['/opt/homebrew/bin/uv','pip','install','--python',str(_repo_path(R, '.venv/bin/python')),'--no-deps','--no-index',str(_repo_path(R, 'wheels', w['filename']))],env=env,check=True)
weights={}
for name,filename,size,hashprefix in [('resnet50','resnet50-11ad3fa6.pth',102540417,'11ad3fa6'),('efficientnet_b0','efficientnet_b0_rwightman-7f5810bc.pth',21444401,'7f5810bc')]:
 weights[name]=fetch('https://download.pytorch.org/models/'+filename,_repo_path(R, 'weights', filename),size,hashprefix);print(json.dumps({name:weights[name]}),flush=True)
license=fetch('https://raw.githubusercontent.com/pytorch/vision/v0.29.0/LICENSE',_repo_path(R, 'licenses/torchvision-BSD-3-Clause.txt'),100000)
write('provenance.json',{'wheel':wheel,'weights':weights,'license':license,'code_license':'BSD-3-Clause; weight-specific/training-data terms distinct, official models documentation retained as URL','weights_notice_url':'https://docs.pytorch.org/vision/stable/models.html','use':'local research only; no redistribution, login, terms acceptance, payment or API','environment':'Separate uv venv. Existing approved dependencies cloned via APFS cp -cR, separate writable inodes; no Torch download or modification of old env. Only torchvision newly installed.','required_dependencies':meta['info']['requires_dist'],'torch':'2.14.0','torchvision':'0.29.0','cost':0})
for n,h in protected.items():assert sha(_repo_path(P, n))==h,n
print(json.dumps({'setup_passed':True,'old_environment_preserved':True}),flush=True)
