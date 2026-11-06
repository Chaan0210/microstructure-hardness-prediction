#!/bin/zsh
PY=../../frameworks/tabpfn-v2/.venv/bin/python; F=../../02-microstructure/0500-heterogeneity/feat-u1dmavgH2Z-train.npz
MODEL=v2 $PY dev600.py $F avg-v2 mps > dev-avg-v2.log 2>&1
IC='{"SOFTMAX_TEMPERATURE":0.9}' $PY dev600.py $F avg-t09 mps > dev-avg-t09.log 2>&1
IC='{"PREPROCESS_TRANSFORMS":[{"name":"none","categorical_name":"ordinal_shuffled","append_original":false,"max_features_per_estimator":768},{"name":"quantile_uni_coarse","categorical_name":"ordinal_shuffled","append_original":true,"max_features_per_estimator":768}]}' $PY dev600.py $F avg-pqu mps > dev-avg-pqu.log 2>&1
IC='{"N_ESTIMATORS":16,"REGRESSION_Y_PREPROCESS_TRANSFORMS":[null,"safepower","quantile_norm"]}' $PY dev600.py $F avg-yq mps > dev-avg-yq.log 2>&1
