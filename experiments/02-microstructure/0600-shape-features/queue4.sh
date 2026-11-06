#!/bin/zsh
PY=../../frameworks/tabpfn-v2/.venv/bin/python
while pgrep -f queue3.sh > /dev/null; do sleep 10; done
$PY dev600.py feat-avgH2Z-gc6-train.npz avg-gc6 mps > dev-avg-gc6.log 2>&1
