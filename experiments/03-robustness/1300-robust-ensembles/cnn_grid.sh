#!/bin/zsh
# quick config search on partition 0, fold 0 (CNN021 on this fold: 12.82)
run() { tag=$1; shift; env "$@" TAG=$tag FOLDS=0 ../../02-microstructure/0220-grain-cellpose/.venv/bin/python cnn_v2.py s 0 0 $EP $LR > cnngrid-$tag.log 2>&1; tail -1 cnngrid-$tag.log >> cnngrid.txt; }
EP=30 LR=1e-4 run r128e30lr1 RES=128
EP=60 LR=1e-4 run r128e60lr1 RES=128
EP=30 LR=3e-4 run r128e30lr3 RES=128
EP=30 LR=1e-4 run r128e30lr1f3 RES=128 FREEZE=3
EP=30 LR=1e-4 run r256e30lr1 RES=256
echo GRIDDONE >> cnngrid.txt
