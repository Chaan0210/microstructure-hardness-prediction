#!/bin/zsh
PY=../../frameworks/tabpfn-v2/.venv/bin/python
$PY dev600.py feat-avgH2Z-g13-train.npz avg-g13 mps > dev-avg-g13.log 2>&1 &
$PY dev600.py feat-avgH2Z-nz3-train.npz avg-nz3 mps > dev-avg-nz3.log 2>&1
wait
