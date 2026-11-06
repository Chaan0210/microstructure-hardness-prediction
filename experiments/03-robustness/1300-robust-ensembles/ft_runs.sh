#!/bin/zsh
F=../../02-microstructure/0800-shading-features/feat-u1dmavgH2Z-anghcor-train.npz
ES=0 PYTHONPATH=../../02-microstructure/0300-cellpose-finetuning ../../frameworks/tabpfn-v2/.venv/bin/python ft_dev.py $F e50lr1 50 1e-5 0 > ft-e50lr1.log 2>&1
ES=0 PYTHONPATH=../../02-microstructure/0300-cellpose-finetuning ../../frameworks/tabpfn-v2/.venv/bin/python ft_dev.py $F e50lr3 50 3e-5 0 > ft-e50lr3.log 2>&1
ES=0 PYTHONPATH=../../02-microstructure/0300-cellpose-finetuning ../../frameworks/tabpfn-v2/.venv/bin/python ft_dev.py $F e150lr1 150 1e-5 0 > ft-e150lr1.log 2>&1
echo done > ft-runs.done
