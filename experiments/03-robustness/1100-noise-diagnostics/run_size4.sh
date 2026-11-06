#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
for t in S4 R4a R4b; do for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  $TP dev900.py ../../03-robustness/1100-noise-diagnostics/feat-${v}-anghcor${t}-train.npz ${v}-anghcor${t} mps > ../../03-robustness/1100-noise-diagnostics/dev-${v}-anghcor${t}.log 2>&1 || echo "FAIL $v $t"
  mv dev-${v}-anghcor${t}.npz dev-${v}-anghcor${t}.json ../../03-robustness/1100-noise-diagnostics/ 2>/dev/null
done; done
echo SIZE4_DONE
