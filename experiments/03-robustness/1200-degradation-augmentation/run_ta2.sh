#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
TP="${TASK_REPO_ROOT}/experiments/frameworks/tabpfn-v2/.venv/bin/python"
cd "${TASK_REPO_ROOT}/experiments/03-robustness/1200-degradation-augmentation"
until grep -q "saved fin-ta" final_ta.log; do sleep 30; done
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do $TP tabaug_dev.py $v deg,deg2,deg3,deg4,deg5,deg6,deg7 mps '../../02-microstructure/0700-directional-features/feat-{v}-angh-train.npz' anghTA > tabaug39-$v.log 2>&1 || echo "FAIL39 $v"; done
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do $TP tabaug_dev.py $v deg,deg2,deg3,deg4,deg5,deg6,deg7 mps '../../02-microstructure/0500-heterogeneity/feat-{v}-train.npz' v35TA > tabaug38-$v.log 2>&1 || echo "FAIL38 $v"; done
echo TA2DONE
