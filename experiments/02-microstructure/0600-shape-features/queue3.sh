#!/bin/zsh
PY=../../frameworks/tabpfn-v2/.venv/bin/python
$PY dev600.py feat-u1dm1szH2Z-g3-train.npz m1-g3 mps > dev-m1-g3.log 2>&1 &
$PY dev600.py feat-u1dm2szH2Z-g3-train.npz m2-g3 mps > dev-m2-g3.log 2>&1
wait
$PY dev600.py feat-u1dmavg4H2Z-g3-train.npz avg4-g3 mps > dev-avg4-g3.log 2>&1 &
$PY dev600.py feat-avgH2Z-g2b-train.npz avg-g2b mps > dev-avg-g2b.log 2>&1
wait
$PY dev600.py feat-avgH2Z-g3b-train.npz avg-g3b mps > dev-avg-g3b.log 2>&1
