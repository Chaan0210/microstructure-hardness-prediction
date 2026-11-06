#!/bin/zsh
for g in size shape pore sp vstd stcoh unet_ph unet_m z ang xsh; do
  SEEDS=0,1,2 PYTHONPATH=../../02-microstructure/0300-cellpose-finetuning ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0300-cellpose-finetuning/dev_eval.py abl/feat-r2-$g-train.npz v3.5 r2-$g mps > abl/log-r2-$g.txt 2>&1 || echo "FAIL $g" >> abl/fail-r2.txt
  mv dev-r2-$g.npz dev-r2-$g.json abl/ 2>/dev/null; echo "$g $(date +%T)" >> abl/progress-r2.txt
done
