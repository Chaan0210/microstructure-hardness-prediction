#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
E="${TASK_REPO_ROOT}/experiments"; CP=$E/02-microstructure/0220-grain-cellpose/.venv/bin/python; TP=$E/frameworks/tabpfn-v2/.venv/bin/python
cd $E/03-robustness/1200-degradation-augmentation
for n in 5 6; do
  $TP make_synth$n.py > deg$n-make.log 2>&1 || { echo "MAKEFAIL $n"; exit 1; }
  $CP synth_feats_g.py train_deg$n > deg$n-ft.log 2>&1 || { echo "FTFAIL $n"; exit 1; }
  $CP unet_feats_sg.py u1 synth train_deg$n > deg$n-u1.log 2>&1 || { echo "U1FAIL $n"; exit 1; }
  for m in m1 m2 h1 m3 u4z1; do $CP synth_unet.py train_deg$n $m > synth-train_deg$n-$m.log 2>&1 || { echo "UNETFAIL $n $m"; exit 1; }; done
  $TP build_rows_g.py train_deg$n > deg$n-rows.log 2>&1 || { echo "ROWSFAIL $n"; exit 1; }
  $CP synth_anghcor.py seg deg$n > deg$n-seg.log 2>&1 && $TP synth_anghcor.py build deg$n > deg$n-build.log 2>&1 || { echo "ANGFAIL $n"; exit 1; }
  echo "DONE deg$n"
done
