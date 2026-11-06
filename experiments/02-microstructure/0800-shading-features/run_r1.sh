#!/bin/zsh
cd "$(dirname "$0")"
until grep -q PCDONE run_pc.log; do sleep 10; done
for t in R1a R1b; do for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  [ -f dev-$v-anghcor$t.npz ] || ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0600-shape-features/dev600.py feat-$v-anghcor$t-train.npz $v-anghcor$t mps 2>&1 | grep -v Warn | tail -1
done; done; echo R1DONE
