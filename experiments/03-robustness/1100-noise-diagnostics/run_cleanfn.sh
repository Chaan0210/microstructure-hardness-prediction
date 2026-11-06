#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# TabPFN-3.5 fitted on clean+mid training rows only (noise < 12 gray), 044 slot features, 4 U-Net variants; official dev partitions/folds/seeds.
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"; TP=../../frameworks/tabpfn-v2/.venv/bin/python
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  TRAIN_NOISE=0,0.0470588 $TP dev900.py ../../02-microstructure/0800-shading-features/feat-${v}-anghcor-train.npz ${v}-anghcor-cleanfn mps > ../../03-robustness/1100-noise-diagnostics/dev-${v}-anghcor-cleanfn.log 2>&1 || echo "FAIL $v"
  mv dev-${v}-anghcor-cleanfn.npz dev-${v}-anghcor-cleanfn.json ../../03-robustness/1100-noise-diagnostics/ 2>/dev/null
done
echo CLEANFN_DONE
