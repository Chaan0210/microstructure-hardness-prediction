"""Serial pre-registered controls and reporting after the neural seed pipeline."""
import json
import subprocess
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]


def run(name,args=(),tabpfn=False):
    py=ROOT/('experiments/tabpfn-v2/.venv/bin/python' if tabpfn else '.venv/bin/python')
    print('START',name,list(args),flush=True)
    with (HERE/f'post-{name}.log').open('a') as log:
        subprocess.run([str(py),'-u',str(HERE/name),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,list(args),flush=True)


def main():
    start=time.monotonic()
    while not (HERE/'pipeline-completion.json').exists():
        if time.monotonic()-start>7200:raise TimeoutError('Neural confirmation not complete')
        time.sleep(10)
    aux=json.loads((HERE/'aux-seed0/summary.json').read_text())
    if aux['addition']['gain']>0 or aux['replacement_vs_deletion']['gain']>0:
        run('cnn_experiment.py',['--context-only'])
        run('verify.py',['--folder',HERE/'context-seed0','--poison'])
    run('spotcheck.py',tabpfn=True)
    run('frozen_heads.py')
    run('seed_mean.py')
    run('adaptive_weights.py')
    run('report.py')
    (HERE/'post-completion.json').write_text(json.dumps(dict(completed=True,seconds=time.monotonic()-start),indent=2)+'\n')
    print('ALL PRESPECIFIED EXPERIMENTS AND CHECKS COMPLETE',flush=True)


if __name__=='__main__':main()
