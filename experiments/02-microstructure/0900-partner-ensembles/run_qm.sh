#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do $TP dev900.py feat-${v}-anghcorqm-train.npz ${v}-anghcorqm mps > dev-${v}-anghcorqm.log 2>&1 || echo "DEVFAIL $v"; done
echo QMDONE
