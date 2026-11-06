from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,subprocess,time,json
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];python=_repo_path(ROOT, 'experiments/grain-cellpose-220/.venv/bin/python');start=time.monotonic()
while not (_repo_path(HERE, 'runs/phase-moments-f0-s89/done.json')).exists():
    if time.monotonic()-start>7200:raise RuntimeError('Earlier study GPU jobs did not finish; no CPSAM job started.')
    time.sleep(5)
commands=[('cpsam-c4-probe.log',['cpsam_rotation_cache.py','--probe','2']),('cpsam-c4-cache.log',['cpsam_rotation_cache.py']),('cpsam-global-c4-f0-s11-e120.log',['cpsam_long_rotation.py','--tag','cpsam-global-c4-f0-s11-e120','--epochs','120'])]
for log,args in commands:
    if log=='cpsam-c4-cache.log':
        r=json.loads((_repo_path(HERE, 'cpsam-prefix-c4/probe.json')).read_text());assert r['peak_RSS']+r['mps_driver']<6*2**30 and max(r['quantization_relative_RMS'])<.001
    print('START',args,flush=True)
    with (_repo_path(HERE, log)).open('w') as stream:subprocess.run([str(python),'-u',str(_repo_path(HERE, args[0])),*args[1:]],cwd=_repo_io(ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('DONE',args,flush=True)
