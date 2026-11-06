#!/bin/zsh
# serial v2 U-Net training (same commands as REPRO-035/038)
PY=../../../.venv/bin/python
$PY phase_unet.py u1 3000 0 > unet-u1.log 2>&1
$PY phase_unet2.py m1 4000 0 0 32 > unet2-m1.log 2>&1
HEAVY=0.5 $PY phase_unet2.py h1 6000 3 0 32 > unet2-h1.log 2>&1
$PY phase_unet2.py m2 4000 1 1 32 > unet2-m2.log 2>&1
$PY phase_unet2.py m3 4000 5 1 32 > unet2-m3.log 2>&1
../../02-microstructure/0220-grain-cellpose/.venv/bin/python phase_unet4.py z1 4000 0 > unet4-z1.log 2>&1
echo "v2 done $(date +%T)" >> done.txt
