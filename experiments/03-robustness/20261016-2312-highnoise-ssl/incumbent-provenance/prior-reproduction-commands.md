# Reproduce this research

Run from `.`.
No installation or network access is needed. All outputs must remain in a new
directory directly below `experiments/`. Existing outputs use exclusive creation
and must not be deleted or overwritten to rerun.

## Exact executed commands

```sh
D=experiments/04-residual-cnn/20261016-2035-plateau-break
.venv/bin/python -B "$D/freeze.py"
.venv/bin/python -B "$D/study.py" selfcheck
.venv/bin/python -B "$D/study.py" extract
.venv/bin/python -B "$D/study.py" h1
.venv/bin/python -B "$D/study.py" h2
.venv/bin/python -B "$D/study.py" h3
.venv/bin/python -B "$D/evaluate.py"
.venv/bin/python -B "$D/verify.py"
.venv/bin/python -B "$D/repair_quantiles.py"
.venv/bin/python -B "$D/consolidate.py"
.venv/bin/python -B "$D/describe.py"
.venv/bin/python -B "$D/reproduce.py" cached
.venv/bin/python -B "$D/reproduce.py" linear
.venv/bin/python -B "$D/reproduce.py" gmm
.venv/bin/python -B "$D/reproduce.py" neural
env REPLAY_DEVICE=mps TABPFN_MODEL_CACHE_SIZE=1 TABPFN_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 \
  experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/reproduce.py" tab
.venv/bin/python -B "$D/reproduce.py" final
.venv/bin/python -B "$D/raw_spot.py"
env TABPFN_MODEL_CACHE_SIZE=1 experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/precision_train.py"
.venv/bin/python -B "$D/preserve.py"
.venv/bin/python -B "$D/manifest.py"
```

The TabPFN replay retains completed member checkpoints and validates their
input hashes. It uses the existing native built-model cache, explicit memory
saving, and only each member's deployed query rows; a few completed original
full-query CPU calls are retained. Explicit float32 is required on MPS to match
CPU precision. On a host without MPS use `REPLAY_DEVICE=cpu`;
this is slower. MPS uses a30% per-process recommended-memory cap. Every deployed
bank aggregate is checked against the original CPU output. None of these runtime
choices modifies model weights, contexts, ensemble seeds or the submission.

The exact executed source for the research scores is locked by
`pre-score-lock.json`. Runtime-only reproduction repairs and their failure logs
are retained. New experiment code contains no historical output destination.

## A clean rerun without altering this record

Choose a new sibling directory, then copy code and the locked protocol only:

```sh
D=experiments/04-residual-cnn/20261016-2035-plateau-break
R=experiments/plateau-replay-20261016-r1
mkdir "$R"
cp "$D"/freeze.py "$D"/study.py "$D"/evaluate.py "$D"/verify.py \
   "$D"/reproduce.py "$D"/raw_spot.py "$D"/repair_quantiles.py "$D"/consolidate.py \
   "$D"/precision_train.py "$D"/preserve.py "$D"/describe.py "$D"/manifest.py "$D"/PRECISION-DIAGNOSTIC.md \
   "$D"/PLAN.md "$D"/pre-score-lock.json "$R/"
```

Use `D=$R` in the commands above. These scripts infer the new output directory
from their own location and keep the project root unchanged. Do not run the
historical builders in their original directories: several write to protected
baseline locations. Final manifest verification is provided separately by
`preserve.py`; copy it too when verifying a fresh rerun.
Run `manifest.py` last after all writers have exited; do not redirect its stdout
to a file inside the directory while that directory is being hashed.

## Settings and saved evidence

- Physical projection directions: NumPy generator2026101601; rotation control:
  2026101602. No hardness labels enter measurement extraction.
- Inner KFold: three folds, seed2026101635+100×outer_fold. Outer folds are the
  original400-image partition stored in `baseline076.npz`.
- Paired bootstrap:10,000 draws, seed2026101636. Auxiliary/calibration description:
  seed2026101637. All uncertainty units are original image IDs.
- Kernel grids, physical scales, masks and advancement rules: `PLAN.md`.
- New final selected models:20 `h1-*.joblib` files, five H3 files in `models/`,
  and ten H2 files in `h2-tie-repair/`. The original H2 models remain in `models/`.
- Final OOF predictions:`final-predictions.npz`; evidence:`final-per-image.csv`;
  authoritative scores:`final-comparison.csv` and `final-results.json`.
- Fit scopes and selected settings: `h1-selection.json`, `h2-selection.json`,
  `h3-selection.json`; source/parameter checks:`verification.json`.
- Baseline SHA inventories:`protected.json`, `protected-upstream-extension.json`.
- Baseline environment snapshots:`environments.json`; no package was upgraded.
- Incumbent assembly:`incumbent-076-cache-replay.csv`, `cache-replay.json`.
- Incumbent model replay:`incumbent-076-model-replay.csv`, `model-replay.json`,
  component replay JSON/NPZs, and `tab-replay-float32/` member predictions.
- Matched training-size precision diagnostic:`precision-train-results.json`,
  `precision-train-oof.npz`, and90 cached contexts under `precision-train/`.
- Raw-pixel saved-z1 inference spot check:`raw-image-spot-replay.json`.

Research used root Python3.13, numpy2.5.3, scipy1.18.1, scikit-learn1.9.1 and
torch2.14.0. TabPFN replay uses the existing separate environment with torch2.14.1
and tabpfn9.1.0. Full package lists, Python patch version and executable paths
are recorded, rather than inferred from these abbreviated versions.

This reproduces the new conditional-head experiments and the incumbent's final
prediction heads from frozen features. It does not retrain all historical image
encoders, regenerate every train/test feature, or reproduce the entire original
training run.
