#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
E="${TASK_REPO_ROOT}/experiments"; CP=$E/02-microstructure/0220-grain-cellpose/.venv/bin/python; TP=$E/frameworks/tabpfn-v2/.venv/bin/python
cd $E/03-robustness/1200-degradation-augmentation
$CP synth_feats_g.py train_deg4 > deg4-ft.log 2>&1 || { echo FTFAIL; exit 1; }
$CP unet_feats_sg.py u1 synth train_deg4 > deg4-u1.log 2>&1 || { echo U1FAIL; exit 1; }
for n in m1 m2 h1 m3 u4z1; do $CP synth_unet.py train_deg4 $n > synth-train_deg4-$n.log 2>&1 || { echo "UNETFAIL $n"; exit 1; }; done
echo FEATSDONE
