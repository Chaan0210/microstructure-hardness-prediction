#!/bin/zsh
# Compliant DTA copies c1..c4 (make_copies_c.py, already generated) -> production noisy-row feature pipeline (REPRO-057 §2/§5 scripts, compliant U-Nets)
R=$(cd $(dirname $0) && pwd); E=$R/experiments; TP=$E/tabpfn-v2/.venv/bin/python; CP=$E/grain-cellpose-220/.venv/bin/python; mkdir -p $R/logs
cd $E/push-1200
for n in ${@:-c1 c2 c3 c4}; do
  $CP synth_feats_g.py train_$n > $R/logs/s2-ft-$n.log 2>&1
  $CP unet_feats_sg.py u1 synth train_$n > $R/logs/s2-u1-$n.log 2>&1
  for m in m1 m2 h1 m3 u4z1; do $CP synth_unet.py train_$n $m > $R/logs/s2-$m-$n.log 2>&1; done
  $TP build_rows_g.py train_$n > $R/logs/s2-rows-$n.log 2>&1
  $CP synth_anghcor.py seg $n > $R/logs/s2-seg-$n.log 2>&1 && $TP synth_anghcor.py build $n > $R/logs/s2-build-$n.log 2>&1
  echo "copies $n done $(date +%T)" >> $R/logs/progress.txt
done
