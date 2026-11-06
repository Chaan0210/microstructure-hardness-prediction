#!/bin/zsh
cd "$(dirname "$0")"
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do [ -f dev-$v-anghcorPC.npz ] || ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0600-shape-features/dev600.py feat-$v-anghcorPC-train.npz $v-anghcorPC mps 2>&1 | grep -v Warn | tail -1; done; echo PCDONE
