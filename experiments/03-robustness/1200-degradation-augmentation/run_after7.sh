#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "${TASK_REPO_ROOT}/experiments/03-robustness/1200-degradation-augmentation"
until [ -f run_deg7.log ] && grep -q "DONE deg7\|FAIL" run_deg7.log; do sleep 30; done
./run_deg8910.sh > run_deg8910.log 2>&1
