#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
E="${TASK_REPO_ROOT}/experiments"; CP=$E/02-microstructure/0220-grain-cellpose/.venv/bin/python; TP=$E/frameworks/tabpfn-v2/.venv/bin/python; PY="${TASK_REPO_ROOT}/.venv/bin/python"
cd $E/02-microstructure/0400-unet-maps
until [ -f unet-u1b.pt ]; do sleep 20; done; sleep 5
cd $E/03-robustness/1200-degradation-augmentation
for sp in train test; do $CP unet_feats_sg.py u1b $sp > ../../03-robustness/1300-robust-ensembles/u1b-feat-$sp.log 2>&1 || echo "FEATFAIL $sp"; done
for s in deg deg2 deg3 deg4; do $CP unet_feats_sg.py u1b synth train_$s > ../../03-robustness/1300-robust-ensembles/u1b-feat-$s.log 2>&1 || echo "SYNFAIL $s"; done
cd $E/03-robustness/1300-robust-ensembles
$PY u1b_diag.py > u1b_diag.log 2>&1
$PY build_u1b.py deg,deg2,deg3,deg4 > u1b_build.log 2>&1 || echo BUILDFAIL
$TP tabU_dev.py u1dmavgH2Z 'feat-{v}-anghcorB-train.npz' none anghcorB > tabB.log 2>&1 || echo TABFAIL
echo U1BDONE
