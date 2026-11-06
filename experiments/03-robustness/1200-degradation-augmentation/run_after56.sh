#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "${TASK_REPO_ROOT}/experiments/03-robustness/1200-degradation-augmentation"
until grep -q "DONE deg6\|FAIL" run_deg56.log; do sleep 30; done
./run_deg7.sh > run_deg7.log 2>&1
