#!/bin/zsh
PY=../../frameworks/tabpfn-v2/.venv/bin/python
$PY dev600.py feat-u1dm1szH2Z-dup-train.npz m1-dup mps > dev-m1-dup.log 2>&1 &
$PY dev600.py feat-u1dm2szH2Z-dup-train.npz m2-dup mps > dev-m2-dup.log 2>&1
wait
$PY dev600.py feat-u1dmavgH2Z-dup-train.npz avg-dup mps > dev-avg-dup.log 2>&1 &
$PY dev600.py feat-u1dmavg4H2Z-dup-train.npz avg4-dup mps > dev-avg4-dup.log 2>&1
wait
