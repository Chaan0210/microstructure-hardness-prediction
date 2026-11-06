#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# wait for the clean-half TTA train masks, pause the chained test segmentation (rerun later if adopted), build features, run 12 dev jobs
cd "${TASK_REPO_ROOT}/experiments/02-microstructure/0900-partner-ensembles"
until [ -f masks-ctta-train.npz ]; do sleep 5; done; sleep 15
pkill -f "segment_ctta.py test"; echo "test segmentation paused"
../../02-microstructure/0220-grain-cellpose/.venv/bin/python ctta_feats.py train > ctta-feats-train.log 2>&1 || { echo FEATFAIL; tail -5 ctta-feats-train.log; exit 1; }
for v in u1dm1sz u1dm2sz u1dmavg u1dmavg4; do
  for suf in "H2Z-ctta" "H2Z-angh-ctta" "H2Z-anghcor-ctta"; do
    f=feat-${v}${suf}-train.npz; [ "$suf" = "H2Z-anghcor-ctta" ] && f=feat-${v}H2Z-anghcor-ctta-train.npz
    ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0600-shape-features/dev600.py $f ${v}${suf} mps > dev-${v}${suf}.log 2>&1 || echo "DEVFAIL $f"
  done
done
echo ALLDONE
