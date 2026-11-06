#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# residual-on-DTA TabPFN slots (d1234 copies), dev: 3 slots x 4 variants (044 u1dmavgH2Z already done)
TP="${TASK_REPO_ROOT}/experiments/frameworks/tabpfn-v2/.venv/bin/python"
cd "${TASK_REPO_ROOT}/experiments/03-robustness/1300-robust-ensembles"
S=deg,deg2,deg3,deg4
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  [ -f dev-$v-anghcorRS4.npz ] || $TP resid_dev.py $v $S mps '../../02-microstructure/0800-shading-features/feat-{v}-anghcor-train.npz' anghcorRS4 > rs4-44-$v.log 2>&1 || echo "FAIL44 $v"
  $TP resid_dev.py $v $S mps '../../02-microstructure/0700-directional-features/feat-{v}-angh-train.npz' anghRS4 > rs4-39-$v.log 2>&1 || echo "FAIL39 $v"
  $TP resid_dev.py $v $S mps '../../02-microstructure/0500-heterogeneity/feat-{v}-train.npz' v35RS4 > rs4-38-$v.log 2>&1 || echo "FAIL38 $v"
done; echo RSDEVDONE
