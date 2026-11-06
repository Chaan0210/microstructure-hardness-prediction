#!/bin/zsh
PY=../../frameworks/tabpfn-v2/.venv/bin/python; F=../../02-microstructure/0500-heterogeneity/feat-u1dmavgH2Z-train.npz
TRAIN_NOISE=0,0.047 $PY dev600.py $F avg-cm mps > dev-avg-cm.log 2>&1
TRAIN_NOISE=0.032,1 $PY dev600.py $F avg-mn mps > dev-avg-mn.log 2>&1
