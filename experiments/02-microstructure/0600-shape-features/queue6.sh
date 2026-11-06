#!/bin/zsh
PY=../../frameworks/tabpfn-v2/.venv/bin/python
$PY dev600.py feat-avgH2Z-hist-train.npz avg-hist mps > dev-avg-hist.log 2>&1 &
$PY dev600.py feat-avgH2Z-rn23-train.npz avg-rn23 mps > dev-avg-rn23.log 2>&1
wait
