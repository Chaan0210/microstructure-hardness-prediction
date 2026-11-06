#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# after r2 training: D4-TTA maps/features (train, test), het columns, label-free diagnostic, noisy-row sets, then 12 dev jobs (after the ctta dev jobs)
E="${TASK_REPO_ROOT}/experiments"; CP=$E/02-microstructure/0220-grain-cellpose/.venv/bin/python; TP=$E/frameworks/tabpfn-v2/.venv/bin/python
until [ -f $E/02-microstructure/0400-unet-maps/unet4-r2.pt ]; do sleep 10; done; sleep 5
cd $E/02-microstructure/0400-unet-maps && for s in train test; do $CP unet4_feats.py r2 $s r2 > unet4-r2-feats-$s.log 2>&1 || { echo "FEATFAIL $s"; exit 1; }; done
cd $E/02-microstructure/0500-heterogeneity && $TP unet_het.py u4r2 > uhet-u4r2.log 2>&1 || { echo HETFAIL; exit 1; }
cd $E/02-microstructure/0900-partner-ensembles && $TP unet_diag.py z1,m1,r2 > r2-diag.txt 2>&1; cat r2-diag.txt
for s in train test; do $TP unet_sets.py r2 replace $s > /dev/null && $TP unet_sets.py r2 avg $s > /dev/null || echo "SETFAIL $s"; done
until grep -q ALLDONE run_ctta_dev.log 2>/dev/null; do sleep 15; done
for v in u1dm1sz u1dm2sz u1dmavg u1dmavg4; do
  for suf in "H2Z-r2replace" "H2Z-angh-r2replace" "H2Z-anghcor-r2replace"; do
    $TP ../../02-microstructure/0600-shape-features/dev600.py feat-${v}${suf}-train.npz ${v}${suf} mps > dev-${v}${suf}.log 2>&1 || echo "DEVFAIL ${v}${suf}"
  done
done
echo R2DONE
