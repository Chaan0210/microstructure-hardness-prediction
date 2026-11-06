#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
TP="${TASK_REPO_ROOT}/experiments/frameworks/tabpfn-v2/.venv/bin/python"
cd "${TASK_REPO_ROOT}/experiments/03-robustness/1200-degradation-augmentation"
for v in u1dm1szH2Z u1dm2szH2Z u1dmavg4H2Z; do $TP tabaug_dev.py $v deg,deg2,deg3,deg4,deg5,deg6,deg7 mps > tabaug-$v.log 2>&1 || echo "FAIL $v"; done
echo TADONE
