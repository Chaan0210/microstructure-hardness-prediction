#!/bin/zsh
# v2 (tone-curve LUT U-Nets): wait for the six weights in push-1400v2, then stage1 -> (GPU: copies c1 c3 c2 c4, then slot-038 dev on MPS)
# || (CPU: slot 039, 044 dev, then final test TabPFN) -> partners (c1..c4) -> blend.
R=$(cd $(dirname $0) && pwd); W=$R/../../1400-compliant-rebuild-v2; E=$R/experiments; TP=$E/tabpfn-v2/.venv/bin/python; mkdir -p $R/logs
until grep -q "v2 done" $W/done.txt 2>/dev/null; do sleep 30; done
cp $W/unet-u1.pt $W/unet2-m1.pt $W/unet2-m2.pt $W/unet2-h1.pt $W/unet2-m3.pt $W/unet4-z1.pt $E/push-400/
echo "v2 weights in place $(date +%T)" >> $R/logs/progress.txt
$R/stage1_features.sh || { echo "STAGE1 FAILED" >> $R/logs/progress.txt; exit 1; }
( $R/stage2_copies.sh c1 c3 c2 c4; $R/stage3_tab.sh mps "038"; echo "GPU lane done $(date +%T)" >> $R/logs/progress.txt ) &
( $R/stage3_tab.sh cpu "039 044"; $R/stage3_tab.sh final; echo "CPU lane done $(date +%T)" >> $R/logs/progress.txt ) &
wait
cd $E/push-1300 && $TP c14_partners.py .. ../push-1200 c1,c2,c3,c4 c14p-v2.npz > $R/logs/partners.log 2>&1 && \
  $TP c14_blend.py .. abl 'c14-h9-{fs}-{s}-{v}' fin-h9-tab-cpu.npz c14p-v2.npz c14b-v2 > $R/logs/blend.log 2>&1
echo "ALL V2 STAGES DONE $(date +%T)" >> $R/logs/progress.txt
