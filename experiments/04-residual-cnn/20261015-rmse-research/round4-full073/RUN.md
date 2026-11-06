# Reproduce the full073 CNN comparison

Run from the repository root. Outputs remain in this directory. No original
artifact, source file, feature cache, submission or other agent's output is
overwritten. Completed teacher/member fits are reused only if their exact
training IDs, targets, configuration and source hashes match.

```bash
.venv/bin/python experiments/04-residual-cnn/20261015-rmse-research/round4-full073/export_anchors.py
.venv/bin/python experiments/04-residual-cnn/20261015-rmse-research/round4-full073/derive_fitter.py
experiments/frameworks/tabpfn-v2/.venv/bin/python -u experiments/04-residual-cnn/20261015-rmse-research/round4-full073/train_teachers.py
.venv/bin/python -u experiments/04-residual-cnn/20261015-rmse-research/round4-full073/continue_pipeline.py
```

The last command may also start while the teacher command is running: it
waits for all30 complete caches, then executes verification and CNN fits
serially with two CPU threads. It never launches another teacher process.
Do not run two teacher commands against the same incomplete node.

Individual model fits can be reproduced with `cnn_experiment.py`:
no option for image-only residual; `--stats` for the mean/SD control;
`--aux` for the frozen auxiliary-objective hypothesis. `--context-only`
provides the image-free control if the auxiliary arm improves. `--repeat 1` uses
the pre-registered second seed. Each run writes per-fold selector and final
CNN checkpoints, all predictions, training curves and inner weights.

`verify.py --folder PATH --poison` checks all30 teacher scopes, the full073
formula, exact checkpoint reload, inner-only coefficients, and a complete
held-out-label poison refit. The verification does not modify the labels.

`PLAN.md`, `AMENDMENT-01.md`, `label-graph.json`, `teacher-provenance.json`
and `environment.json` define the scope and configuration. These are
development-validation experiments. Official061 and073 remain the
reproducible submission references; the previous `../reproduce_best.py`
reconstructs their existing CSVs without changing them. A new submission
requires a separately documented full500 refit and a frozen final rule;
this validation pipeline does not silently promote a development winner.

After the CNN pipeline and any conditional context-only control finish:

```bash
experiments/frameworks/tabpfn-v2/.venv/bin/python experiments/04-residual-cnn/20261015-rmse-research/round4-full073/spotcheck.py
.venv/bin/python experiments/04-residual-cnn/20261015-rmse-research/round4-full073/frozen_heads.py
.venv/bin/python experiments/04-residual-cnn/20261015-rmse-research/round4-full073/seed_mean.py
.venv/bin/python experiments/04-residual-cnn/20261015-rmse-research/round4-full073/adaptive_weights.py
.venv/bin/python experiments/04-residual-cnn/20261015-rmse-research/round4-full073/report.py
```

The additional weight-only and joint-blend comparison is defined in
`AMENDMENT-02.md`. It uses the existing internal predictions and performs
no new model training or outer-label coefficient fit.

`AMENDMENT-03.md` defines the cached ImageNet/MicroNet frozen-CNN head
comparison. It preserves study205's EDF32 capacity and compares direct-y
versus honest residual targets. No new encoder inference or gradient
training occurs in that secondary study.

The post-analysis noise-regime experiment is reproducible with
`noise_calibration.py`. It chooses the amount of pooling by leave-one-out
validation inside B only. `matched_controls.py` first completes the second
seeds of the two controls, then regenerates fixed means and both calibration
comparisons. See `AMENDMENT-05.md` and `AMENDMENT-06.md` for this exploratory
sequence and its limitations.

The separate full500 candidate pipeline is:

```bash
experiments/frameworks/tabpfn-v2/.venv/bin/python -u experiments/04-residual-cnn/20261015-rmse-research/round4-full073/final_teachers.py
.venv/bin/python -u experiments/04-residual-cnn/20261015-rmse-research/round4-full073/final_model.py fit
.venv/bin/python -u experiments/04-residual-cnn/20261015-rmse-research/round4-full073/final_model.py infer
.venv/bin/python experiments/04-residual-cnn/20261015-rmse-research/round4-full073/verify_final.py
```

Fitting blocks all test image/feature reads. Inference uses frozen weights
and training normalization, and reads no training labels or training pixels.
The two experimental CSVs are written under `final500/`; neither is uploaded
or copied over a shared numbered submission. `FINAL500.md` documents the
fixed production coefficient aggregation and the absence of a new scored
500-trained validation model. Existing completed fits are verified and reused.
