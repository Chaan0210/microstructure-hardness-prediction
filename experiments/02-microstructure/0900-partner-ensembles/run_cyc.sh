#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
# after cycgan training: translate the 249 clean train images once (K=1), export PNGs to push-400/synth/train_cyc, then the production noisy-row
# feature pipeline (u1 U-Net first for the realism check, ft-a1s250 4-view cellpose features, multi-task/z U-Nets), synthetic rows, dev (038 slot).
E="${TASK_REPO_ROOT}/experiments"; CP=$E/02-microstructure/0220-grain-cellpose/.venv/bin/python; TP=$E/frameworks/tabpfn-v2/.venv/bin/python
cd $E/02-microstructure/0900-partner-ensembles/cyc; until grep -q "^DONE" train.log; do sleep 20; done
$TP cycgan.py apply 1 > apply.log 2>&1 || { echo APPLYFAIL; exit 1; }
"$TP" - <<'PY'
import json, numpy as np
from pathlib import Path
from PIL import Image
D = np.load('trans-k1.npz'); out = Path('../../../02-microstructure/0400-unet-maps/synth/train_cyc'); out.mkdir(parents=True, exist_ok=True); log = {}
for i, im in zip(D['ids'], D['images'][0]):
    Image.fromarray(im).save(out / f'{i}.png'); a = im.astype(float); hh = (a[::2, ::2] - a[::2, 1::2] - a[1::2, ::2] + a[1::2, 1::2]) / 2; log[str(i)] = dict(measured_noise=float(np.median(np.abs(hh)) / .6745))
json.dump(log, open('../../../02-microstructure/0400-unet-maps/synth/train_cyc_params.json', 'w'), indent=0); print('exported', len(log))
PY
cd $E/02-microstructure/0400-unet-maps && $CP unet_feats_scyc.py u1 synth > unet-u1-synthcyc.log 2>&1 || { echo U1FAIL; exit 1; }
echo U1DONE
$CP synth_feats_cyc.py > synth-ftCY.log 2>&1 || { echo FTFAIL; exit 1; }
cd $E/02-microstructure/0500-heterogeneity && for n in m1 m2 h1 m3 u4z1; do $CP synth_unet.py train_cyc $n > synth-unet-cyc-$n.log 2>&1 || { echo "UNETFAIL $n"; exit 1; }; done
$TP build_synth_rows_cyc.py > build-synthcyc.log 2>&1 || { echo BUILDFAIL; tail -3 build-synthcyc.log; exit 1; }
echo ROWSDONE
cd $E/02-microstructure/0300-cellpose-finetuning && for fl in 1 0; do for v in u1dm1sz u1dm2sz u1dmavg u1dmavg4; do
  SYN=../../02-microstructure/0500-heterogeneity/synthrows-train_cyc-${v}H2Z.npz FLAG=$fl $TP dev_eval_syn.py ../../02-microstructure/0500-heterogeneity/feat-${v}H2Z-train.npz v3.5 ${v}H2Z-syncycf$fl mps > dev-${v}H2Z-syncycf$fl.log 2>&1 || echo "DEVFAIL $v $fl"; done; done
echo CYCDONE
