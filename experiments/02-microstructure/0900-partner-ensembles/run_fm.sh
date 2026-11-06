#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# TabPFN-family partners on the 044-slot feature sets (4 variants x 3 seeds, dev protocol)
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
for M in v3.5-fast v2.6 v3 v2.5 v2.5-real v2.5-small; do
  for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do MODEL=$M $TP dev900.py ../../02-microstructure/0800-shading-features/feat-${v}-anghcor-train.npz ${v}-anghcor-$M mps > dev-${v}-anghcor-$M.log 2>&1 || echo "DEVFAIL $M $v"; done
  echo "MODELDONE $M"
done
echo FMDONE
