#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
E="${TASK_REPO_ROOT}/experiments"; CP=$E/02-microstructure/0220-grain-cellpose/.venv/bin/python; TP=$E/frameworks/tabpfn-v2/.venv/bin/python
cd $E/02-microstructure/0300-cellpose-finetuning; until grep -q "^DONE" ft-R300.log 2>/dev/null || grep -q "Traceback" ft-R300.log 2>/dev/null; do sleep 15; done
grep -q Traceback ft-R300.log && { echo TRAINFAIL; tail -5 ft-R300.log; exit 1; }
$CP seg_ft.py ft-R300.pt train noisy 1 > segft-R300-train-n1.log 2>&1 || { echo SEGFAIL; tail -5 segft-R300-train-n1.log; exit 1; }
cd $E/02-microstructure/0900-partner-ensembles && $TP ft_diag.py a1s250,H300,R300
echo R300DIAGDONE
