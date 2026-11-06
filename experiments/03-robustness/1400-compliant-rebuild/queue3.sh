#!/bin/zsh
# serial queue (two concurrent MPS trainings ran ~4x slower each under memory pressure)
PY=../../../.venv/bin/python
while kill -0 50161 2>/dev/null; do sleep 15; done
kill -CONT 56790; while kill -0 56790 2>/dev/null; do sleep 15; done
$PY phase_unet2.py m2 4000 1 1 32 > unet2-m2.log 2>&1
$PY phase_unet2.py m3 4000 5 1 32 > unet2-m3.log 2>&1
../../02-microstructure/0220-grain-cellpose/.venv/bin/python phase_unet4.py z1 4000 0 > unet4-z1.log 2>&1
echo "serial queue done $(date +%T)" >> unets-done.txt
