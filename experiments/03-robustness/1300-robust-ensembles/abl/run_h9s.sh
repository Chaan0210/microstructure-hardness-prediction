#!/bin/zsh
for tag in h9s h9r; do for fs in nb full; do for v in u1dmavg u1dm2sz u1dm1sz u1dmavg4; do
  SEEDS=0,1,2 PYTHONPATH=../../02-microstructure/0300-cellpose-finetuning ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0300-cellpose-finetuning/dev_eval.py abl/feat-$tag-$fs-044-$v-train.npz v3.5 $tag-$fs-044-$v mps > abl/log-$tag-$fs-$v.txt 2>&1 || echo "FAIL $tag $fs $v" >> abl/fail-h9s.txt
  mv dev-$tag-$fs-044-$v.npz dev-$tag-$fs-044-$v.json abl/ 2>/dev/null; echo "$tag $fs $v $(date +%T)" >> abl/progress-h9s.txt
done; done; done
