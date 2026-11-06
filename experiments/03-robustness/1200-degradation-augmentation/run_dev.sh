#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# usage: run_dev.sh TAG [TAG ...]  -> dev-<v>-anghcor<TAG>.npz for the 4 U-Net variants (044 slot, TabPFN-3.5 x3 seeds, official 3 partitions)
TP="${TASK_REPO_ROOT}/experiments/frameworks/tabpfn-v2/.venv/bin/python"
cd "${TASK_REPO_ROOT}/experiments/03-robustness/1200-degradation-augmentation"
for t in "$@"; do for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  [ -f dev-$v-anghcor$t.npz ] || $TP ../../02-microstructure/0900-partner-ensembles/dev900.py feat-$v-anghcor$t-train.npz $v-anghcor$t mps > dev-$v-anghcor$t.log 2>&1 || echo "FAIL $t $v"
done; echo "DONE $t"; done
