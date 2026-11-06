#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
CP="${TASK_REPO_ROOT}/experiments/02-microstructure/0220-grain-cellpose/.venv/bin/python"
cd "${TASK_REPO_ROOT}/experiments/03-robustness/1200-degradation-augmentation"
$CP zsnet.py infer A,B synth:train_deg2 zs-synth-deg2 > zs-infer-deg2.log 2>&1
$CP zsnet.py infer A,B synth:train_deg3 zs-synth-deg3 > zs-infer-deg3.log 2>&1
$CP zsnet.py infer A,B train zs-train > zs-infer-train.log 2>&1
$CP zsnet.py infer A,B test zs-test > zs-infer-test.log 2>&1
echo ZSDONE
