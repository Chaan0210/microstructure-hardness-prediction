#!/bin/zsh
# test predictions (CPU, deterministic) for shade-family components; cids 046c12, 046c18, 046zs, 046zsp (not submission candidates by themselves)
cd "$(dirname "$0")/../../02-microstructure/0300-cellpose-finetuning"
fin() { cid=$1; tag=$2; ../../frameworks/tabpfn-v2/.venv/bin/python finalize_multi.py $cid 0.75 ../../02-microstructure/0800-shading-features/combos.py cpu \
  v3.5@../../02-microstructure/0800-shading-features/feat-u1dm1szH2Z-angh$tag-train.npz@../../02-microstructure/0800-shading-features/feat-u1dm1szH2Z-angh$tag-test.npz \
  v3.5@../../02-microstructure/0800-shading-features/feat-u1dm2szH2Z-angh$tag-train.npz@../../02-microstructure/0800-shading-features/feat-u1dm2szH2Z-angh$tag-test.npz \
  v3.5@../../02-microstructure/0800-shading-features/feat-u1dmavgH2Z-angh$tag-train.npz@../../02-microstructure/0800-shading-features/feat-u1dmavgH2Z-angh$tag-test.npz \
  v3.5@../../02-microstructure/0800-shading-features/feat-u1dmavg4H2Z-angh$tag-train.npz@../../02-microstructure/0800-shading-features/feat-u1dmavg4H2Z-angh$tag-test.npz > ../../02-microstructure/0800-shading-features/finalize-$cid.log 2>&1; echo "done $cid"; }
fin 046c12 c12 & fin 046c18 c18 & wait
fin 046zs corZs & fin 046zsp corZsP & wait
echo FINDONE
