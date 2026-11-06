"""Complete the pre-registered second-seed, equally calibrated controls."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[4];PY=_repo_path(ROOT, '.venv/bin/python')


def run(name,args,log):
    with (_repo_path(HERE, log)).open('a') as out:
        subprocess.run([str(PY),'-u',str(_repo_path(HERE, name)),*map(str,args)],cwd=_repo_io(ROOT),stdout=out,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,list(map(str,args)),flush=True)


if __name__=='__main__':
    for family,flag in [('stats','--stats'),('context','--context-only')]:
        run('cnn_experiment.py',[flag,'--repeat','1'],f'{family}-seed1.log')
        run('verify.py',['--folder',_repo_path(HERE, f'{family}-seed1')],f'{family}-seed1-verification.log')
    run('seed_mean.py',[],'matched-seed-mean.log')
    run('adaptive_weights.py',[],'matched-adaptive-weights.log')
    run('noise_calibration.py',[],'matched-noise-calibration.log')
    print('MATCHED CONTROLS COMPLETE',flush=True)
