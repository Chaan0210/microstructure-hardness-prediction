#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
for v in u1dm1sz u1dm2sz u1dmavg u1dmavg4; do for suf in "H2Z-r2avg" "H2Z-angh-r2avg" "H2Z-anghcor-r2avg"; do
  $TP ../../02-microstructure/0600-shape-features/dev600.py feat-${v}${suf}-train.npz ${v}${suf} mps > dev-${v}${suf}.log 2>&1 || echo "DEVFAIL ${v}${suf}"; done; done
$TP cmp045all.py r2avg; echo R2AVGDONE
