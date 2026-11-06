#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
until grep -q QM2DONE run_qm2.log 2>/dev/null; do sleep 20; done
for s in 44 38 39 44orig; do $TP final_qm.py tab $s > final-qm-tab$s.log 2>&1 || echo "FINALFAIL $s"; done
echo FINALQMDONE
