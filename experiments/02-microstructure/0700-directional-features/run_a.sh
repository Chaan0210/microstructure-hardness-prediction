cd ../../02-microstructure/0600-shape-features
for v in u1dm1szH2Z u1dm2szH2Z u1dmavg4H2Z; do ../../frameworks/tabpfn-v2/.venv/bin/python dev600.py ../../02-microstructure/0700-directional-features/feat-$v-hetel-train.npz $v-hetel 2>&1 | grep -v -i warn; echo "== $v done"; done
