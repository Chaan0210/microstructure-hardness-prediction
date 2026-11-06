# Reproduce the relational image-learning study

Project root: `.`.
Use the existing root `.venv`; nothing is installed or downloaded. Outputs are
confined to this experiment. Do not run historical submission builders in place.

```sh
D=experiments/03-robustness/20261016-highnoise-relational
.venv/bin/python -B "$D/protect.py" freeze
.venv/bin/python -B "$D/experiment.py" selfcheck
.venv/bin/python -B "$D/experiment.py" prepare
.venv/bin/python -B "$D/experiment.py" probe
.venv/bin/python -B "$D/experiment.py" train
.venv/bin/python -B "$D/evaluate.py"
.venv/bin/python -B "$D/mechanisms.py"
.venv/bin/python -B "$D/verify.py"
.venv/bin/python -B "$D/reproduce.py"
.venv/bin/python -B "$D/protect.py" verify
```

After all writers have exited, `.venv/bin/python -B "$D/manifest.py"` hashes the
closed deliverables. Do not redirect that command's stdout into this directory
while it is being hashed. `finalize.py` closes the supplied report once; it is
not a training stage and is not needed to reproduce the predictions.

Run stages sequentially. `train` resumes only complete fits after checking their
contracts; incomplete directories are preserved for diagnosis, not silently
deleted or overwritten. Most result files use exclusive creation. To rerun from
scratch, create a **new sibling under `experiments/`** and copy only the Python
source files, `PLAN.md` and `DIAGNOSTICS.md` there. The scripts derive their new
output path from their own location. Do not copy or remove old model outputs.

The existing protection inventories and frozen baseline inputs are referenced
by absolute project-relative paths in the scripts. `protect.py freeze` first
verifies the previous076 inventory, then adds082's measured result and the
additional inner076 anchors. It records current Git status and all installed
root-environment versions. No Git history is changed.

If and only if the predeclared primary advancement rule passes, run the same
three arms at the second fixed seed, then evaluate their equal seed means:

```sh
.venv/bin/python -B "$D/experiment.py" train --repeat 1
.venv/bin/python -B "$D/evaluate.py" --repeats 0 1
.venv/bin/python -B "$D/mechanisms.py" --repeats 0 1
```

Do not choose the better seed. A second seed is a stochastic sensitivity check
on the same labels, not independent data confirmation. Full500 fitting or a new
submission is not performed automatically by these scripts.

## Exact algorithm

- Main seeds: `2026101651 + 100*fold`; repeat offset100000. Same initialization,
  sample order, two D4 draws and dropout RNG consumption in all matched arms.
- Shuffled relational-target assignment: `seed + 7919`, training sources only.
- Image/target mean and SD: fit on actual A213 or T320 training rows only.
- Pixel encoder:70,129 parameters, native256, GroupNorm/SiLU, spatial moments,
  32-dimensional learned embedding and a scalar hardness head.
- Neighborhood target width:0.5 training-target SD. Cosine-logit temperature0.2.
  Neighborhood loss weight0 forMSE,1 forREL andSHUFFLED. All retain scalar MSE.
- Training:40epochs,32sources/two views, AdamW.002/weight decay.01, cosine→.0001,
  clip5, last-five-state mean. Four-view CPU float32 inference. No early stopping.
- New coefficient: one bounded OLS value in[0,.25], fitted only on eligible
  high-noise B107 rows. Existing082 coefficient procedure is reproduced first.
- Bootstrap:10000 image-ID draws, seed2026101652. Conditional coefficient
  bootstrap:2000 draws, seed2026101660+fold, old082 coefficient held fixed.

## Artifacts and interpretation

- `starting-state.json`, `protected.json`: incumbent/environment/code provenance.
- `baseline-verification.json`, `anchors/`: exact082 coefficient reconstruction
  and source-isolated inner076/G021 predictions.
- `code-lock.json`, `analysis-code-lock.json`, `mechanism-code-lock.json`: recipes
  recorded before new comparative scores.
- `models/seed0/`:30 completed model checkpoints, normalization, source IDs,
  training curves, gradients, source/view counts, query predictions and embeddings.
- `resource-probe/`:discarded two-epoch training-only fits, no validation score.
- `results-0.json`, `comparison-0.csv`, `oof-0.npz`, `per-image-0.csv`: primary
  nested comparison and controls, all five folds and paired uncertainty.
- `mechanisms-0.json`: gated spread decomposition and held-out embedding geometry;
  descriptive only, not an OOF-derived deployable adjustment.
- `verification-0.json`, `label-poison-replay/`: every checkpoint reload plus a
  fresh40-epoch refit with all outside labels changed and dataset reads blocked.
- `reproduction/`: freshly reassembled076 and saved-CNN082 inference; no original
  output path is written. Both CSV identities are checked exactly.
- `preservation.json`: final full SHA256 check of protected files and Git state.

All500 labels have prior exposure. New encoder learning and calibration exclude
outer V, but inherited082 auxiliary representations are frozen historical inputs.
The study does not certify a new independent holdout or reproduce all historical
training. It must not be used to assert an expected private-leaderboard score.
