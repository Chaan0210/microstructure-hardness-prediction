#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
E="${TASK_REPO_ROOT}/experiments"; TP=$E/frameworks/tabpfn-v2/.venv/bin/python; PY="${TASK_REPO_ROOT}/.venv/bin/python"
cd $E/03-robustness/1300-robust-ensembles
until grep -q -E "DONE deg12|FAIL" run_deg12.log; do sleep 20; done
grep -q FAIL run_deg12.log && { echo PIPEFAIL; exit 1; }
cd $E/03-robustness/1200-degradation-augmentation
$PY ../../03-robustness/1300-robust-ensembles/realism_sets.py deg12 deg4 > ../../03-robustness/1300-robust-ensembles/realism-deg12.log 2>&1
NOISEMIN=12 $PY synaug_ridge2.py anghcor deg,deg2,deg3,deg4,deg12 sr-A-d1234_12 > ../../03-robustness/1300-robust-ensembles/sr-d1234_12.log 2>&1 || echo SRFAIL1
NOISEMIN=12 $PY synaug_ridge2.py anghcor deg,deg2,deg3,deg12 sr-A-d123_12 > ../../03-robustness/1300-robust-ensembles/sr-d123_12.log 2>&1 || echo SRFAIL2
cd $E/03-robustness/1300-robust-ensembles
$TP tabU_dev.py u1dmavgH2Z '../../02-microstructure/0800-shading-features/feat-{v}-anghcor-train.npz' deg12 anghcorT12 > tab12.log 2>&1 || echo TABFAIL
echo AFTER12DONE
