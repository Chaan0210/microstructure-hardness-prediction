# Reproduce the bounded080 follow-up

From repository root, using only the already installed local environments:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/04-residual-cnn/20261016-candidate-080-followup/study.py audit
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/04-residual-cnn/20261016-candidate-080-followup/replay.py profile
PYTHONDONTWRITEBYTECODE=1 experiments/frameworks/tabpfn-v2/.venv/bin/python experiments/04-residual-cnn/20261016-candidate-080-followup/study.py extend
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/04-residual-cnn/20261016-candidate-080-followup/replay.py replay
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/04-residual-cnn/20261016-candidate-080-followup/replay.py resplit
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/04-residual-cnn/20261016-candidate-080-followup/subgroups.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/04-residual-cnn/20261016-candidate-080-followup/version_chain.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/04-residual-cnn/20261016-candidate-080-followup/finish.py
```

All writes are inside this directory. Existing full073 helpers are imported
read-only, with bytecode writes disabled and a local Matplotlib cache. The
teacher output root is redirected only in the current process. Two original
anchors0/1 are fixed. Teacher extension fitting has a60-minute bound, CPU2
threads, unchanged model/seed/feature recipes, and eight overlap checks per
anchor. Completed new member caches are verified and reused.

Calibration/evaluation splits are107/80 with five ShuffleSplit repeats,
seed2026101608. The64 alignment permutations use seeds2026101611+1000*anchor
+permutation. Bootstrap seed is2026101610. No public metric is passed to a
fit or calibration routine. No test labels, external data, installation,
submission generation/upload, Git mutation or other-process control occurs.

`continue_checks.py` can wait for the two extensions and then run the fixed
replay/resplit checks serially. `subgroups.py` recomputes the same split and
permutation outputs, verifies identical primary values, and decomposes by
the already fixed noise12 boundary; it is not an additional fitted model.

The initial combined-query legacy inference differed at floating-point level
from its originalB/V batches. Original batch layouts were restored and checked;
the small difference is recorded separately. No old result/model was edited.

These are conditional diagnostics on previously explored images. E3 keeps
A-only predictors frozen, so its RMSE is not a replacement full080 CV score.
The former100-image holdout is not independent. No global production
coefficient is evaluated on old OOF as if it were nested validation.
