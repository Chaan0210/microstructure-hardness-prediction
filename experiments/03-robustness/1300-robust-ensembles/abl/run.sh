#!/bin/zsh
# usage: run.sh SEEDS group...   (from push-1300)
S=$1; shift
for g in "$@"; do
  SEEDS=$S PYTHONPATH=../../02-microstructure/0300-cellpose-finetuning ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0300-cellpose-finetuning/dev_eval.py abl/feat-drop-$g-train.npz v3.5 abl-$g-s${S//,/} mps > abl/log-$g-s${S//,/}.txt 2>&1 || echo "FAIL $g" >> abl/fail.txt
  mv dev-abl-$g-s${S//,/}.npz dev-abl-$g-s${S//,/}.json abl/ 2>/dev/null
  echo "$g done $(date +%T)" >> abl/progress.txt
done
