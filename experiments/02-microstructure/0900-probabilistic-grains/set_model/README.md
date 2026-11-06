# Hardness-supervised grain-set experiment

All writes stay in this directory. Inputs are the parent experiment's original-train object bank and fixed 400-image legacy-fold comparator. No legacy code is edited. No synthetic/augmented rows, test training, per-crop label replication, or outer-validation early stopping is used.

The fixed control is Ridge on area-weighted raw object means and standard deviations, concatenated with raw global features. Ridge alpha is selected within training images with a four-fold pipeline. The small set model adds a context-conditioned nonlinear encoder and area-weighted mean/standard-deviation pooling to that fixed Ridge skip. It minimizes one direct hardness loss per image. All grains are included. Object ordering is irrelevant; absolute angle and position remain input features.

The frozen design in `run.py:CONFIG` uses one seed recipe. An inner 80/20 image split selects the epoch count, including epoch zero (the Ridge skip). The final outer-fold model refits normalization, Ridge, and neural parameters on all 320 training images. The 80 outer-validation images affect neither fitting nor stopping. The other 100 train images in the 500-image bank do not participate in this CV.

Commands from repository root, after the root agent authorizes this stage:

```sh
.venv/bin/python experiments/02-microstructure/0900-probabilistic-grains/set_model/run.py --mode benchmark
.venv/bin/python experiments/02-microstructure/0900-probabilistic-grains/set_model/run.py --mode train --device auto
.venv/bin/python experiments/02-microstructure/0900-probabilistic-grains/set_model/run.py --mode verify --device cpu
```

`auto` chooses CPU or MPS using training-only forward/backward timing, without validation scores. Only one MPS process should run at a time. Two Torch CPU threads are used. Each fold saves its model, training/validation image IDs, inner epoch histories, selected epoch, Ridge alpha, reload comparison, and inference permutation/padding checks. The first fold also checks that changing all labels outside the fitting scope leaves normalization, Ridge predictions, and an actual neural training update unchanged. Padding is only an inference audit and never training data.

Results include direct predictors, a fixed 20% predictor blend with 045, and a blend preserving the existing 021 share. Existing failure fallback is applied. Reusing the legacy folds gives a conditional comparison, not a fresh independent validation estimate.
