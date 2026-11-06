#!/bin/zsh
# memory-safe queue: at most 2 concurrent U-Net trainings (16GB machine; each ~4GB with MPS)
PY=../../../.venv/bin/python
if [ "$1" = "A" ]; then
  while kill -0 50160 2>/dev/null; do sleep 20; done          # u1
  $PY phase_unet2.py m1 4000 0 0 32 > unet2-m1.log 2>&1
  $PY phase_unet2.py m2 4000 1 1 32 > unet2-m2.log 2>&1
  $PY phase_unet2.py m3 4000 5 1 32 > unet2-m3.log 2>&1
  echo "done A2" >> unets-done.txt
else
  while kill -0 50161 2>/dev/null; do sleep 20; done          # h1
  ../../02-microstructure/0220-grain-cellpose/.venv/bin/python phase_unet4.py z1 4000 0 > unet4-z1.log 2>&1
  echo "done Z" >> unets-done.txt
fi
