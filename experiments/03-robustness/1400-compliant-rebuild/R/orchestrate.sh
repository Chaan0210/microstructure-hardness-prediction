#!/bin/zsh
# Wait for the six compliant U-Nets, then run all stages (two GPU jobs max; CPU TabPFN lane in parallel).
R=$(cd $(dirname $0) && pwd); W=$R/..; E=$R/experiments; mkdir -p $R/logs
while true; do
  ok=1; for f in unet-u1.pt unet2-m1.pt unet2-m2.pt unet2-h1.pt unet2-m3.pt unet4-z1.pt; do [ -f $W/$f ] || ok=0; done
  if [ $ok = 1 ] && ! pgrep -f "phase_unet" > /dev/null; then break; fi; sleep 30
done
cp $W/unet-u1.pt $W/unet2-m1.pt $W/unet2-m2.pt $W/unet2-h1.pt $W/unet2-m3.pt $W/unet4-z1.pt $E/push-400/
echo "weights in place $(date +%T)" >> $R/logs/progress.txt
$R/stage1_features.sh || { echo "STAGE1 FAILED" >> $R/logs/progress.txt; exit 1; }
# GPU lane: copies, then slot 038 dev on MPS ; CPU lane: slots 039 044 dev on CPU, then final test predictions on CPU
( $R/stage2_copies.sh c1 c2 c3 c4; $R/stage3_tab.sh mps "038"; echo "GPU lane done $(date +%T)" >> $R/logs/progress.txt ) &
( $R/stage3_tab.sh cpu "039 044"; $R/stage3_tab.sh final; echo "CPU lane done $(date +%T)" >> $R/logs/progress.txt ) &
wait; echo "ALL STAGES DONE $(date +%T)" >> $R/logs/progress.txt
