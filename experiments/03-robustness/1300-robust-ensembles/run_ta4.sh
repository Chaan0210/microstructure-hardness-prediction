#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# TA slots with the 061 copy set (deg,deg2,deg3,deg4) for the sim-augmented log-Ridge column: dev (3 slots x 4 variants) then final (3 slots)
TP="${TASK_REPO_ROOT}/experiments/frameworks/tabpfn-v2/.venv/bin/python"
cd "${TASK_REPO_ROOT}/experiments/03-robustness/1200-degradation-augmentation"
S=deg,deg2,deg3,deg4
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  $TP tabaug_dev.py $v $S mps '../../02-microstructure/0800-shading-features/feat-{v}-anghcor-train.npz' anghcorTA4 > ../../03-robustness/1300-robust-ensembles/ta4-44-$v.log 2>&1 || echo "FAIL44 $v"
  $TP tabaug_dev.py $v $S mps '../../02-microstructure/0700-directional-features/feat-{v}-angh-train.npz' anghTA4 > ../../03-robustness/1300-robust-ensembles/ta4-39-$v.log 2>&1 || echo "FAIL39 $v"
  $TP tabaug_dev.py $v $S mps '../../02-microstructure/0500-heterogeneity/feat-{v}-train.npz' v35TA4 > ../../03-robustness/1300-robust-ensembles/ta4-38-$v.log 2>&1 || echo "FAIL38 $v"
done; echo DEVDONE
$TP final_ta.py $S mps '../../02-microstructure/0800-shading-features/feat-{v}-anghcor-{split}.npz' fin-ta4-44 > ../../03-robustness/1300-robust-ensembles/fin-ta4-44.log 2>&1 || echo FAILF44
$TP final_ta.py $S mps '../../02-microstructure/0700-directional-features/feat-{v}-angh-{split}.npz' fin-ta4-39 > ../../03-robustness/1300-robust-ensembles/fin-ta4-39.log 2>&1 || echo FAILF39
$TP final_ta.py $S mps '../../02-microstructure/0500-heterogeneity/feat-{v}-{split}.npz' fin-ta4-38 > ../../03-robustness/1300-robust-ensembles/fin-ta4-38.log 2>&1 || echo FAILF38
echo ALLDONE
