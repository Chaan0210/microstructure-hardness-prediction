#!/bin/zsh
for sd in 0 1; do env RES=128 FREEZE=3 TAG=-r128f3 ../../02-microstructure/0220-grain-cellpose/.venv/bin/python cnn_v2.py s 0 $sd 30 1e-4 >> cnnp0.log 2>&1; done
echo P0DONE >> cnnp0.log
