#!/bin/zsh
# usage: run_ctx.sh CTXMAX slot variant...   (nb features)
C=$1; s=$2; shift 2
for v in "$@"; do
  CTXMAX=$C SEEDS=0,1,2 PYTHONPATH=../../02-microstructure/0300-cellpose-finetuning ../../frameworks/tabpfn-v2/.venv/bin/python dev_eval_ctx.py abl/feat-nb-$s-$v-train.npz v3.5 ctx$C-$s-$v mps > abl/log-ctx$C-$s-$v.txt 2>&1 || echo "FAIL ctx $C $s $v" >> abl/fail-ctx.txt
  mv dev-ctx$C-$s-$v.npz dev-ctx$C-$s-$v.json abl/ 2>/dev/null; echo "ctx$C $s $v $(date +%T)" >> abl/progress-ctx.txt
done
