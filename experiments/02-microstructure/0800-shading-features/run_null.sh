#!/bin/zsh
# null distribution: 039 + one unscreened N(0,1) column (seeds 810..860), 4 variants each. arg1 = device, args = control indices
cd "$(dirname "$0")"; dev=$1; shift
for k in "$@"; do for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  [ -f dev-$v-anghrnd$k.npz ] && continue
  ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0600-shape-features/dev600.py feat-$v-anghrnd$k-train.npz $v-anghrnd$k $dev 2>&1 | grep -v Warn | tail -1
done; echo "done rnd$k $(date +%T)"; done
