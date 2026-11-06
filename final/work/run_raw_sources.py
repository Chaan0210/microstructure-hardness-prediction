from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
base=HERE.parent
python=ROOT/'experiments/grain-cellpose-220/.venv/bin/python'
for seed in [1811,1823]:
    with (base/f'verification/source-raw-{seed}.log').open('w') as log:
        subprocess.run([str(python),'-u',str(base/'package/source_features.py'),'--data',str(ROOT/'data'),'--out',str(base/'verification/source-raw'),'--seed',str(seed),'--split','test'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('RAW SOURCE COMPLETE',seed,flush=True)
