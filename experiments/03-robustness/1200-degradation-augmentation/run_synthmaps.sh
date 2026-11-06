#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
CP="${TASK_REPO_ROOT}/experiments/02-microstructure/0220-grain-cellpose/.venv/bin/python"
cd "${TASK_REPO_ROOT}/experiments/03-robustness/1200-degradation-augmentation"
for st in train_deg2 train_deg3 train_deg; do for n in m1 m2 h1 m3; do
  $CP synth_unet.py $st $n > synth-$st-$n.log 2>&1 || echo "FAIL $st $n"
done; done
echo ALLDONE
