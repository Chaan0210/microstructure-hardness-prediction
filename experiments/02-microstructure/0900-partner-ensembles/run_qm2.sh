#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# after run_qm.sh (044 slot): 038 slot (push-500 features) and 039 slot (push-700 angh) on the aligned features
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
until grep -q QMDONE run_qm.log 2>/dev/null; do sleep 20; done
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do $TP dev900.py feat-${v}-qm-train.npz ${v}-qm mps > dev-${v}-qm.log 2>&1 || echo "DEVFAIL38 $v"; done
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do $TP dev900.py feat-${v}-anghqm-train.npz ${v}-anghqm mps > dev-${v}-anghqm.log 2>&1 || echo "DEVFAIL39 $v"; done
echo QM2DONE
