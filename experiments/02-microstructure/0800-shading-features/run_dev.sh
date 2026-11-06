#!/bin/zsh
# dev600 (3 partitions x 5 folds x seeds 0,1,2, TabPFN-3.5, mps) for 039 + one column: cor (shading-corrected), raw, rnd (control)
cd "$(dirname "$0")"
for s in cor raw rnd; do for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  [ -f dev-$v-angh$s.npz ] && continue
  echo "start $s $v $(date +%T)"
  ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0600-shape-features/dev600.py feat-$v-angh$s-train.npz $v-angh$s mps 2>&1 | grep -v Warn
done; done
echo ALLDONE
