#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
for t in DA75 DA60; do
  $TP dev900.py ../../03-robustness/1100-noise-diagnostics/feat-u1dmavgH2Z-anghcor${t}-train.npz u1dmavgH2Z-anghcor${t} mps > ../../03-robustness/1100-noise-diagnostics/dev-u1dmavgH2Z-anghcor${t}.log 2>&1 || echo "FAIL $t"
  mv dev-u1dmavgH2Z-anghcor${t}.npz dev-u1dmavgH2Z-anghcor${t}.json ../../03-robustness/1100-noise-diagnostics/ 2>/dev/null
done
echo DEGALL_DONE
