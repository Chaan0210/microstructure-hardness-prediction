#!/bin/zsh
while pgrep -f run_h9s.sh > /dev/null; do sleep 20; done
for fs in nb full; do for v in u1dmavg u1dm2sz u1dm1sz u1dmavg4; do
  SEEDS=0,1,2 PYTHONPATH=../../02-microstructure/0300-cellpose-finetuning ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0300-cellpose-finetuning/dev_eval.py abl/feat-h9g-$fs-044-$v-train.npz v3.5 h9g-$fs-044-$v mps > abl/log-h9g-$fs-$v.txt 2>&1 || echo "FAIL $fs $v" >> abl/fail-h9g.txt
  mv dev-h9g-$fs-044-$v.npz dev-h9g-$fs-044-$v.json abl/ 2>/dev/null; echo "h9g $fs $v $(date +%T)" >> abl/progress-h9g.txt
done; done
