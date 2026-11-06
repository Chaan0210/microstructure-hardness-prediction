cd ../../02-microstructure/0600-shape-features
for v in u1dm2szH2Z u1dmavg4H2Z; do ../../frameworks/tabpfn-v2/.venv/bin/python dev600.py ../../02-microstructure/0700-directional-features/feat-$v-angh-train.npz $v-angh 2>&1 | grep -v -i warn; echo "== $v angh done"; done
../../frameworks/tabpfn-v2/.venv/bin/python dev600.py ../../02-microstructure/0700-directional-features/feat-u1dm2szH2Z-ang-train.npz u1dm2szH2Z-ang 2>&1 | grep -v -i warn; echo "== m2 ang done"
