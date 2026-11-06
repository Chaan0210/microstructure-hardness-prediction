#!/bin/zsh
# Production copy feature chain (push-1400/R/stage2_copies.sh / push-1300/run_deg12.sh order) for one rendered set W/synth/train_<n>,
# all outputs in W/feat. usage: run_set.sh <n>   (expects W/synth/train_<n>/*.png and W/synth/train_<n>_params.json)
W=$(cd $(dirname $0)/.. && pwd); E=$W/../..; TP=$E/frameworks/tabpfn-v2/.venv/bin/python; CP=$E/02-microstructure/0220-grain-cellpose/.venv/bin/python; n=$1; L=$W/logs
cd $W/pipe
$CP synth_feats_g.py train_$n > $L/$n-ft.log 2>&1 || { echo "FTFAIL $n" >> $L/progress.txt; exit 1; }
$CP unet_feats_sg.py u1 synth train_$n > $L/$n-u1.log 2>&1 || { echo "U1FAIL $n" >> $L/progress.txt; exit 1; }
for m in m1 m2 h1 u4z1; do $CP synth_unet.py train_$n $m > $L/$n-$m.log 2>&1 || { echo "UNETFAIL $n $m" >> $L/progress.txt; exit 1; }; done
$TP build_rows_g.py train_$n > $L/$n-rows.log 2>&1 || { echo "ROWSFAIL $n" >> $L/progress.txt; exit 1; }
$CP synth_anghcor.py seg $n > $L/$n-seg.log 2>&1 && $TP synth_anghcor.py build $n > $L/$n-build.log 2>&1 || { echo "ANGFAIL $n" >> $L/progress.txt; exit 1; }
echo "DONE $n $(date +%T)" >> $L/progress.txt
