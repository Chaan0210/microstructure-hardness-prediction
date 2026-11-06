#!/bin/zsh
# Compliant feature rebuild (push-1400 mirror R). Same scripts as REPRO-035/038/045 + push-1300 H9/nb; only the U-Net weights differ
# (retrained with mask-free degradation). Run from R/. Logs in R/logs/.
set -e
R=$(cd $(dirname $0) && pwd); mkdir -p $R/logs; E=$R/experiments
PY=$R/.venv/bin/python; TP=$E/tabpfn-v2/.venv/bin/python; CP=$E/grain-cellpose-220/.venv/bin/python
cd $E/push-400
for s in train test; do
  OUTDIR=. NOMAPS=1 $PY unet_feats.py u1 $s > $R/logs/u1-$s.log 2>&1
  $PY unet2_feats.py m1 $s m1 > $R/logs/m1-$s.log 2>&1
  LA_MU=5.6343302726745605 LA_SD=0.9692867994308472 $PY unet2_feats.py m2 $s m2 > $R/logs/m2-$s.log 2>&1
  $PY unet2_feats.py h1 $s h1 > $R/logs/h1-$s.log 2>&1
  LA_MU=5.6343302726745605 LA_SD=0.9692867994308472 $PY unet2_feats.py m3 $s m3 > $R/logs/m3-$s.log 2>&1
  $CP unet4_feats.py z1 $s z1 > $R/logs/z1-$s.log 2>&1
  $TP pipeline_final.py assemble $s --out . > $R/logs/assemble-$s.log 2>&1
  echo "unet+assemble $s done" >> $R/logs/progress.txt
done
cd $E/push-500
for n in m1 m2 h1 m3; do $TP unet_het.py $n > $R/logs/het-$n.log 2>&1; done
$TP make_het_sets.py het4_u,het4_u_mx H2 > $R/logs/hetsets.log 2>&1
$TP add_z.py z1 z_f25,z_f50 H2 H2Z > $R/logs/addz.log 2>&1
cd $E/push-700
for s in train test; do $TP angle_feats.py $s > $R/logs/angle-$s.log 2>&1; done
$TP build_sets.py angh u1dm1szH2Z u1dm2szH2Z u1dmavgH2Z u1dmavg4H2Z > $R/logs/angh.log 2>&1
cd $E/push-800
for s in train test; do $CP shade_feat.py $s > $R/logs/shade-$s.log 2>&1; $TP add_cols.py cor $s cor > $R/logs/addcols-$s.log 2>&1; done
cd $E/push-1300
$TP cumul_ext.py > $R/logs/cumul.log 2>&1
$TP build_nb_h9.py $E $E/push-1300/abl > $R/logs/nbh9.log 2>&1
echo "stage1 done" >> $R/logs/progress.txt
