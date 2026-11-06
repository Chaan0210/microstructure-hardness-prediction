#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
E="${TASK_REPO_ROOT}/experiments"; TP=$E/frameworks/tabpfn-v2/.venv/bin/python; PY="${TASK_REPO_ROOT}/.venv/bin/python"
cd $E/03-robustness/1300-robust-ensembles
until grep -q -E "DONE deg11|FAIL" run_deg11.log; do sleep 20; done
grep -q FAIL run_deg11.log && { echo PIPEFAIL; exit 1; }
cd $E/03-robustness/1200-degradation-augmentation
$PY ../../03-robustness/1300-robust-ensembles/realism_sets.py deg11 deg4 > ../../03-robustness/1300-robust-ensembles/realism-deg11.log 2>&1
NOISEMIN=12 $PY synaug_ridge2.py anghcor deg,deg2,deg3,deg4,deg11 sr-A-d1234_11 > ../../03-robustness/1300-robust-ensembles/sr-d1234_11.log 2>&1 || echo SRFAIL1
NOISEMIN=12 $PY synaug_ridge2.py anghcor deg,deg2,deg3,deg11 sr-A-d123_11 > ../../03-robustness/1300-robust-ensembles/sr-d123_11.log 2>&1 || echo SRFAIL2
cd $E/03-robustness/1300-robust-ensembles
$TP tabU_dev.py u1dmavgH2Z '../../02-microstructure/0800-shading-features/feat-{v}-anghcor-train.npz' deg11 anghcorT11 > tab11.log 2>&1 || echo TABFAIL
echo AFTER11DONE
