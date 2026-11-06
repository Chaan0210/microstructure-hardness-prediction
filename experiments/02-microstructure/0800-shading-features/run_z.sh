#!/bin/zsh
# after training: U-Net features (train/test) for z1s, z1c; feature sets; dev runs (MPS)
cd "$(dirname "$0")"
for n in z1s z1c; do for s in train test; do
  [ -f ../../02-microstructure/0400-unet-maps/unet4-$n-$s.npz ] || (cd ../../02-microstructure/0400-unet-maps && NOMAPS=1 ../../../.venv/bin/python unet4_feats.py $n $s $n 2>&1 | grep -v Warn | tail -1)
done; done
for s in train test; do
  ../../frameworks/tabpfn-v2/.venv/bin/python swap_z.py z1s corZs z $s; ../../frameworks/tabpfn-v2/.venv/bin/python swap_z.py z1c corZc z $s
  ../../frameworks/tabpfn-v2/.venv/bin/python swap_z.py z1s corZsP zp $s; ../../frameworks/tabpfn-v2/.venv/bin/python swap_z.py z1c corZcP zp $s
done
for t in corZs corZc corZsP corZcP; do for v in u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z; do
  [ -f dev-$v-angh$t.npz ] && continue
  ../../frameworks/tabpfn-v2/.venv/bin/python ../../02-microstructure/0600-shape-features/dev600.py feat-$v-angh$t-train.npz $v-angh$t mps 2>&1 | grep -v Warn | tail -1
done; echo "done $t $(date +%T)"; done
echo ZDONE
