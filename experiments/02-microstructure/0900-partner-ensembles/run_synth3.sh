#!/bin/zsh
TASK_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
E="${TASK_REPO_ROOT}/experiments"; CP=$E/02-microstructure/0220-grain-cellpose/.venv/bin/python; TP=$E/frameworks/tabpfn-v2/.venv/bin/python
cd $E/02-microstructure/0400-unet-maps && $CP synth_feats3.py > synth-ftC.log 2>&1 || { echo FTFAIL; exit 1; }
$CP unet_feats_s3.py u1 synth > unet-u1-synth3.log 2>&1 || { echo U1FAIL; exit 1; }
cd $E/02-microstructure/0500-heterogeneity && for n in m1 m2 h1 m3 u4z1; do $CP synth_unet.py train_deg3 $n > synth-unet-deg3-$n.log 2>&1 || { echo "UNETFAIL $n"; exit 1; }; done
$TP build_synth_rows3.py > build-synth3.log 2>&1 || { echo BUILDFAIL; tail -3 build-synth3.log; exit 1; }
echo ROWSDONE
cd $E/02-microstructure/0300-cellpose-finetuning && for fl in 1 0; do for v in u1dm1sz u1dm2sz u1dmavg u1dmavg4; do
  SYN=../../02-microstructure/0500-heterogeneity/synthrows-train_deg3-${v}H2Z.npz FLAG=$fl $TP dev_eval_syn.py ../../02-microstructure/0500-heterogeneity/feat-${v}H2Z-train.npz v3.5 ${v}H2Z-syn3f$fl mps > dev-${v}H2Z-syn3f$fl.log 2>&1 || echo "DEVFAIL $v $fl"; done; done
echo SYNTH3DONE
