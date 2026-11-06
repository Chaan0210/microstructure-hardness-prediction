"""Train/infer only after the final cross-fitted teacher stage has closed."""
import json
import subprocess
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]


def run(script,args,log):
    print('START',script,args,flush=True)
    with (HERE/log).open('a') as out:
        subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(HERE/script),*args],cwd=ROOT,
                       stdout=out,stderr=subprocess.STDOUT,check=True)
    print('DONE',script,args,flush=True)


if __name__=='__main__':
    start=time.monotonic()
    while not (HERE/'final500/teachers-complete.json').exists():
        if time.monotonic()-start>7200:raise TimeoutError('Final teacher stage not complete')
        time.sleep(10)
    run('final_model.py',['fit'],'final-model-fit.log')
    run('final_model.py',['infer'],'final-model-infer.log')
    run('verify_final.py',[],'final-independent-verification.log')
    (HERE/'final500/pipeline-complete.json').write_text(json.dumps(dict(complete=True,seconds=time.monotonic()-start,uploaded=False),indent=2)+'\n')
    print('FINAL500 CANDIDATES COMPLETE AND VERIFIED',flush=True)
