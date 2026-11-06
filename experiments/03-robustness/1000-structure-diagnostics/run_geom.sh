#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# after geomnet g1 training: inference (train/test, 8-view TTA), geometry columns, noisy-row measurement check, 044-slot dev with the 13 columns
cd "${TASK_REPO_ROOT}/experiments/03-robustness/1000-structure-diagnostics"; CP=../../02-microstructure/0220-grain-cellpose/.venv/bin/python; TP=../../frameworks/tabpfn-v2/.venv/bin/python
until grep -q "^DONE" geomnet-g1.log 2>/dev/null || grep -q Traceback geomnet-g1.log 2>/dev/null; do sleep 20; done
grep -q Traceback geomnet-g1.log && { echo TRAINFAIL; exit 1; }
$CP geomnet.py infer g1 > geomnet-g1-infer.log 2>&1 || { echo INFERFAIL; exit 1; }
$TP geom_feats.py g1 > geom-feats-g1.log 2>&1 || { echo FEATFAIL; exit 1; }
$TP geom_check.py g1 > geom-check-g1.txt 2>&1; cat geom-check-g1.txt
echo CHECKDONE
cd ../../02-microstructure/0900-partner-ensembles && for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do $TP dev900.py ../../03-robustness/1000-structure-diagnostics/feat-${v}-anghcorGg1-train.npz ${v}-anghcorGg1 mps > dev-${v}-anghcorGg1.log 2>&1 || echo "DEVFAIL $v"; done
echo GEOMDEVDONE
