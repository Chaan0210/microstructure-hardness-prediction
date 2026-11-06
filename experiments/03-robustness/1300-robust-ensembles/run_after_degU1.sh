#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
E="${TASK_REPO_ROOT}/experiments"; TP=$E/frameworks/tabpfn-v2/.venv/bin/python; PY="${TASK_REPO_ROOT}/.venv/bin/python"
cd $E/03-robustness/1300-robust-ensembles
until grep -q -E "DONE degU1|FAIL" run_degU.log; do sleep 20; done
grep -q FAIL run_degU.log && { echo PIPEFAIL; exit 1; }
cd $E/03-robustness/1200-degradation-augmentation
NOISEMIN=12 $PY synaug_ridge2.py anghcor deg,deg2,deg3,deg4,degU1 sr-A-d1234U1 > ../../03-robustness/1300-robust-ensembles/sr-d1234U1.log 2>&1 || echo SRFAIL1
NOISEMIN=12 $PY synaug_ridge2.py anghcor degU1 sr-A-U1only > ../../03-robustness/1300-robust-ensembles/sr-U1only.log 2>&1 || echo SRFAIL2
cd $E/03-robustness/1300-robust-ensembles
$TP tabU_dev.py u1dmavgH2Z '../../02-microstructure/0800-shading-features/feat-{v}-anghcor-train.npz' degU1 anghcorTU1 > tabU1.log 2>&1 || echo TABFAIL
echo AFTERU1DONE
