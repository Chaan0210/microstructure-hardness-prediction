#!/bin/zsh
PY=../../frameworks/tabpfn-v2/.venv/bin/python
$PY dev600.py feat-avgH2Z-r99-train.npz avg-r99 mps > dev-avg-r99.log 2>&1 &
$PY dev600.py feat-avgH2Z-r97-train.npz avg-r97 mps > dev-avg-r97.log 2>&1
wait
