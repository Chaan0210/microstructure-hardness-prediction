#!/bin/zsh
cd "$(dirname "$0")"
for s in train test; do [ -f ../../02-microstructure/0400-unet-maps/unet4-z1p-$s.npz ] || (cd ../../02-microstructure/0400-unet-maps && NOMAPS=1 ../../../.venv/bin/python unet4_feats.py z1p $s z1p 2>&1 | grep -v Warn | tail -1); done
for s in train test; do ../../frameworks/tabpfn-v2/.venv/bin/python swap_z.py z1p corZp z $s; done
until grep -q ZDONE run_z.log; do sleep 10; done
for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  [ -f dev-$v-anghcorZp.npz ] && continue
  ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0600-shape-features/dev600.py feat-$v-anghcorZp-train.npz $v-anghcorZp mps 2>&1 | grep -v Warn | tail -1
done; echo ZPDONE
