#!/bin/zsh
for s in 038 039 044; do for v in u1dm1sz u1dm2sz u1dmavg u1dmavg4; do
  SEEDS=0,1,2 PYTHONPATH=../../02-microstructure/0300-cellpose-finetuning ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0300-cellpose-finetuning/dev_eval.py abl/feat-nb-$s-$v-train.npz v3.5 nb-$s-$v mps > abl/log-nb-$s-$v.txt 2>&1 || echo "FAIL $s $v" >> abl/fail-nb.txt
  mv dev-nb-$s-$v.npz dev-nb-$s-$v.json abl/ 2>/dev/null; echo "$s $v $(date +%T)" >> abl/progress-nb.txt
done; done
