#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# TabICL and TabDPT on the 044-slot feature sets (CPU, parallel with the MPS TabPFN jobs)
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; FMP=../../frameworks/tabular-fm/.venv/bin/python
for M in tabicl tabdpt; do
  for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do OMP_NUM_THREADS=4 $FMP fm_dev.py $M ../../02-microstructure/0800-shading-features/feat-${v}-anghcor-train.npz ${v}-anghcor-$M > dev-${v}-anghcor-$M.log 2>&1 || echo "DEVFAIL $M $v"; done
  echo "MODELDONE $M"
done
echo FM2DONE
