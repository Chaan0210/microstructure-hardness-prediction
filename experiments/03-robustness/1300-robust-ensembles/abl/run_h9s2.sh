#!/bin/zsh
while pgrep -f run_h9g.sh > /dev/null; do sleep 20; done
for s in 038 039; do for fs in nb full; do for v in u1dmavg u1dm2sz u1dm1sz u1dmavg4; do
  SEEDS=0,1,2 PYTHONPATH=../../02-microstructure/0300-cellpose-finetuning ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0300-cellpose-finetuning/dev_eval.py abl/feat-h9s2-$fs-$s-$v-train.npz v3.5 h9s2-$fs-$s-$v mps > abl/log-h9s2-$fs-$s-$v.txt 2>&1 || echo "FAIL $s $fs $v" >> abl/fail-h9s2.txt
  mv dev-h9s2-$fs-$s-$v.npz dev-h9s2-$fs-$s-$v.json abl/ 2>/dev/null; echo "$s $fs $v $(date +%T)" >> abl/progress-h9s2.txt
done; done; done
