#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
until grep -q RESIDDONE run_resid.log; do sleep 10; done
for m in v2 v3; do for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do MODEL=$m $TP dev900.py ../../02-microstructure/0800-shading-features/feat-${v}-anghcor-train.npz ${v}-anghcor$m mps > dev-${v}-anghcor$m.log 2>&1 || echo "DEVFAIL $v $m"; done; done
echo VERDONE
