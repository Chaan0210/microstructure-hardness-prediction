#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do STRICT_QM=1 $TP dev900.py ../../02-microstructure/0800-shading-features/feat-${v}-anghcor-train.npz ${v}-anghcorqs mps > dev-${v}-anghcorqs.log 2>&1 || echo "DEVFAIL44 $v"; done
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do STRICT_QM=1 $TP dev900.py ../../02-microstructure/0500-heterogeneity/feat-${v}-train.npz ${v}-qs mps > dev-${v}-qs.log 2>&1 || echo "DEVFAIL38 $v"; done
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do STRICT_QM=1 $TP dev900.py ../../02-microstructure/0700-directional-features/feat-${v}-angh-train.npz ${v}-anghqs mps > dev-${v}-anghqs.log 2>&1 || echo "DEVFAIL39 $v"; done
echo QSDONE
