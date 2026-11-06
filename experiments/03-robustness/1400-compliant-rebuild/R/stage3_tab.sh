#!/bin/zsh
# TabPFN block on the compliant h9 sets. usage: stage3_tab.sh DEVICE "SLOTS"  (dev, official 3 partitions x 5 folds)  |  stage3_tab.sh final
# dev_eval.py resolves data relative to the cwd (two levels below the root) -> run from push-1300; outputs are tag-named (no collisions between lanes).
R=$(cd $(dirname $0) && pwd); E=$R/experiments; TP=$E/tabpfn-v2/.venv/bin/python; mkdir -p $R/logs
cd $E/push-1300
if [ "$1" = "final" ]; then $TP final_h9_tab.py > $R/logs/s3-final.log 2>&1; echo "tab final done $(date +%T)" >> $R/logs/progress.txt; exit; fi
dev=$1; for s in ${=2}; do for fs in nb full; do for v in u1dm1sz u1dm2sz u1dmavg u1dmavg4; do
  SEEDS=0,1,2 PYTHONPATH=../push-300 $TP ../push-300/dev_eval.py abl/feat-h9-$fs-$s-$v-train.npz v3.5 c14-h9-$fs-$s-$v $dev > $R/logs/s3-dev-$fs-$s-$v.log 2>&1
  mv dev-c14-h9-$fs-$s-$v.npz dev-c14-h9-$fs-$s-$v.json abl/ 2>/dev/null
  echo "tab dev $dev $s $fs $v $(date +%T)" >> $R/logs/progress.txt
done; done; done
