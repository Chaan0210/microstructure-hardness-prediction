#!/bin/zsh
# Compliant U-Net retraining (same commands as push-400/REPRO-035.md, mask-free degradation)
PY=../../../.venv/bin/python
if [ "$1" = "A" ]; then
  $PY phase_unet.py u1 3000 0 > unet-u1.log 2>&1
  $PY phase_unet2.py m1 4000 0 0 32 > unet2-m1.log 2>&1
  $PY phase_unet2.py m2 4000 1 1 32 > unet2-m2.log 2>&1
  $PY phase_unet2.py m3 4000 5 1 32 > unet2-m3.log 2>&1
else
  HEAVY=0.5 $PY phase_unet2.py h1 6000 3 0 32 > unet2-h1.log 2>&1
fi
echo done $1 >> unets-done.txt
