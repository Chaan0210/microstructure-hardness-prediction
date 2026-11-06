#!/bin/zsh
# wait for chain B (h1) then train compliant z1 (REPRO-038 command)
while ! grep -q "done B" unets-done.txt 2>/dev/null; do sleep 30; done
../../02-microstructure/0220-grain-cellpose/.venv/bin/python phase_unet4.py z1 4000 0 > unet4-z1.log 2>&1
echo done Z >> unets-done.txt
