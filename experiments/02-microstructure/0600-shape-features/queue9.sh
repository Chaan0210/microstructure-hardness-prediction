#!/bin/zsh
PY=../../frameworks/tabpfn-v2/.venv/bin/python
$PY dev600.py feat-m1-rom-train.npz m1-rom mps > dev-m1-rom.log 2>&1 &
$PY dev600.py feat-m2-rom-train.npz m2-rom mps > dev-m2-rom.log 2>&1
wait
$PY dev600.py feat-avg-rom-train.npz avg-rom mps > dev-avg-rom.log 2>&1 &
$PY dev600.py feat-avg4-rom-train.npz avg4-rom mps > dev-avg4-rom.log 2>&1
wait
