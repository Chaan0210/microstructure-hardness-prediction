# Reproduction and retained work

Run from the project root. Existing061/073 and the previous recommended CSV
are read-only inputs. All new outputs stay in this directory. No command uploads.

## Current recommended CSV: unchanged

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidates-078-080/make_submission.py
```

The file is `../078-080-update-20261016/candidate-076-plus-frozen-cnn.csv`.
This study did not create a replacement: no completed candidate passed its
promotion/stability gate. The previous report contains its training/inference
reproduction and the distinction between measured and algebraically implied
Public scores.

## Corrected baseline and cheap probes

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/probe.py baseline
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/probe.py infer
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/probe.py evaluate
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/confidence.py
```

These preserve the original400 images/folds. The baseline reuses verified H9
Tab OOF and unchanged073 components; it refits only076's high-domain linear
partners with source-grouped internal alpha selection.

## New small CNNs and stochastic seed confirmation

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/stream_cnn.py train --repeat 0
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/stream_cnn.py evaluate --repeats 0
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/lowmem_repeat.py
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/control_repeat.py evaluate
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/ensemble_eval.py
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/verify_control.py
```

The first command fits both raw+raw and raw+Gaussian at seed offset0. Only the
raw+raw control is repeated at offset100000. Each outer fold has A213/B107
coefficient selection followed by T320/V80 evaluation; teacher residuals are
`y-T073`, where T073 is the complete non021 branch normalized by.84, from
honest nested073 predictions. Calibration targets the full073 ensemble on B;
outer comparisons use strict076. New students have69,572 parameters,40epochs,
AdamW.002/weight decay.01,cosine schedule,batch32,D4,auxiliary weight.2,
last-five-state averaging,four-view inference. CPU1,1GiB peak check.

`lowmem_repeat.py` reuses completed models and changes only allocation of the
normalized image array. Every normalized pixel is checked against the original
expression. All source/derived hashes and checks are in `lowmem-derivation.json`
and `lowmem-reproduction.json`. Original high-memory failure records are in
`pre-lowmem-verification/`; the failed log is retained. Do not reinterpret it
as a score-based stop. Checkpoints/predictions/curves/IDs are under `stream/`.

## Calibration order and frozen-model role checks

```sh
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/calibration_order.py all
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/finalize.py
```

All coefficients use internal calibration labels only. The extra role check
reuses strict073 A-only teachers and infers the small CNNs on their187 excluded
rows. It uses anchors0/1/4 and five107/80role splits at seed2026101608. Losses are
aggregated by342originalIDs,not treated as1200independent observations.

## Incomplete actual076-target pilot — resume only with enough memory

This is not a completed experiment or a new OOF result. Three teacher bundles
and252 individual Tab predictions are retained. The large teacher stage exited137
after memory pressure; no076-target CNN was trained. Other jobs were not modified.

```sh
env TABPFN_MODEL_CACHE_SIZE=1 experiments/frameworks/tabpfn-v2/.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/retarget_teachers.py --folds 0 1
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/check_teachers.py
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/retarget_cnn.py train --folds 0 1
.venv/bin/python experiments/04-residual-cnn/20261016-candidate-076-complement/retarget_cnn.py evaluate --folds 0 1
```

The native built-model cache was verified to reproduce four varied cases exactly.
It preserves the model/seed/estimator count and reduces rebuilding; it does not
make the model small. The teacher stage has a70minute per-invocation limit and
may still need substantial memory. Do not launch the subsequent CNN stage until
`teachers076-complete-01.json` exists and teacher verification passes. Existing
completed models are reused only when their contracts match. Read the locked
pilot gate in `RETARGET-PLAN.md` before any expansion beyond these two folds.

Current outputs: `REPORT.md`, `comparison.csv`, `fold-metrics.csv`, `decision.json`,
and detailed result JSONs. Source/model hashes are in `protected.json`, the
per-model verification records, and `delivery-hashes.json`.
