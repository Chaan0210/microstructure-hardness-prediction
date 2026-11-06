#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do LOGY=1 $TP dev900.py ../../02-microstructure/0800-shading-features/feat-${v}-anghcor-train.npz ${v}-anghcorlogy mps > dev-${v}-anghcorlogy.log 2>&1 || echo "DEVFAIL $v"; done
echo LOGYDONE
