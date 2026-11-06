cd ../../02-microstructure/0600-shape-features
for v in u1dmavgH2Z u1dm1szH2Z; do ../../frameworks/tabpfn-v2/.venv/bin/python dev600.py ../../02-microstructure/0700-directional-features/feat-$v-angh-train.npz $v-angh 2>&1 | grep -v -i warn; echo "== $v angh done"; done
../../frameworks/tabpfn-v2/.venv/bin/python dev600.py ../../02-microstructure/0700-directional-features/feat-u1dm1szH2Z-ang-train.npz u1dm1szH2Z-ang 2>&1 | grep -v -i warn; echo "== m1 ang done"
