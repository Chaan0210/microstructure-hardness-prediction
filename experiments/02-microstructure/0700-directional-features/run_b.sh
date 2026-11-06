cd ../../02-microstructure/0600-shape-features
for s in 1 2 3; do ../../frameworks/tabpfn-v2/.venv/bin/python dev600.py ../../02-microstructure/0700-directional-features/feat-u1dmavgH2Z-ctrl$s-train.npz avg-ctrl$s 2>&1 | grep -v -i warn; echo "== ctrl$s done"; done
