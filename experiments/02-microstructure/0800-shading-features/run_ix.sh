#!/bin/zsh
cd "$(dirname "$0")"
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  [ -f dev-$v-anghcorIX.npz ] && continue
  ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0600-shape-features/dev600.py feat-$v-anghcorIX-train.npz $v-anghcorIX mps 2>&1 | grep -v Warn | tail -1
done; echo IXDONE
