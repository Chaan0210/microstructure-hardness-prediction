"""Complete the pre-registered second-seed, equally calibrated controls."""
import subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];PY=ROOT/'.venv/bin/python'


def run(name,args,log):
    with (HERE/log).open('a') as out:
        subprocess.run([str(PY),'-u',str(HERE/name),*map(str,args)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,list(map(str,args)),flush=True)


if __name__=='__main__':
    for family,flag in [('stats','--stats'),('context','--context-only')]:
        run('cnn_experiment.py',[flag,'--repeat','1'],f'{family}-seed1.log')
        run('verify.py',['--folder',HERE/f'{family}-seed1'],f'{family}-seed1-verification.log')
    run('seed_mean.py',[],'matched-seed-mean.log')
    run('adaptive_weights.py',[],'matched-adaptive-weights.log')
    run('noise_calibration.py',[],'matched-noise-calibration.log')
    print('MATCHED CONTROLS COMPLETE',flush=True)
